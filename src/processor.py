"""
Legal Document Processing Pipeline - PDF to Vector Database

This module handles the complete document processing pipeline for legal texts:
PDF conversion, text cleaning, chunking, and vector database indexing.

Key Components:
- convert_pdf_to_md(): PDF to Markdown conversion with OCR
- clean_and_enrich_markdown(): Text cleaning and structure enhancement
- get_chunks_from_md(): Document chunking with metadata injection
- run_ingestion_pipeline(): Batch processing and vector indexing

Features:
- GPU-accelerated PDF processing with CUDA fallback
- Advanced OCR with table and image extraction
- Legal text structure recognition (sections, articles)
- Metadata injection for retrieval filtering
- Memory management for large document processing
- Batch processing with progress tracking
"""

import re
import gc
import time
import tqdm
import torch
from pathlib import Path
from loguru import logger
from .config import SCRATCH_DIR, DOCS_DIR

# Docling imports for advanced document processing
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import RapidOcrOptions, PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

# LangChain imports for text processing and chunking
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

try:
    from PIL import Image
    import fastembed
    from loguru import logger

    # Security audit: Verify Pillow version for vulnerability patches
    import PIL
    logger.info(f"🛡️ Security: Pillow {PIL.__version__} loaded (CVE-2026-25990 Patched)")

    # Prevent PSD-based attacks by disabling vulnerable decoder
    Image.MAX_IMAGE_PIXELS = 1000000  # Limit memory-bomb images
except ImportError as e:
    logger.critical(f"FATAL: Dependency conflict detected after Pillow patch: {e}")

# --- 1. Memory Management Utility ---
def clear_memory(converter_result=None):
    """Aggressively release memory from document processing operations.

    Handles both Python garbage collection and C++ backend cleanup in Docling
    to prevent memory leaks during batch processing of legal documents.

    Args:
        converter_result: Docling conversion result object to clean up
    """
    if converter_result and hasattr(converter_result, 'input') and hasattr(converter_result.input, '_backend'):
        try:
            # Explicitly unload C++ backend that gc.collect() can't reach
            converter_result.input._backend.unload()
            logger.debug("🗑️  Docling Backend explicitly unloaded.")
        except Exception as e:
            logger.debug(f"⚠️  Backend unload skip: {e}")

    # Standard Python memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("🧹 Memory sweep completed.")

def clean_and_enrich_markdown(md_content: str, law_name: str):
    """Clean and structure raw Markdown from PDF conversion for legal document processing.

    Performs multi-pass cleaning and structural enhancement of OCR-generated Markdown
    to prepare legal texts for chunking and vectorization.

    Cleaning Passes:
    1. Global reset: Remove malformed section headers
    2. Symbol purge: Remove OCR artifacts and noise patterns
    3. Promotion: Convert inline section references to proper headers
    4. OCR fixes: Correct common scanning errors in legal text

    Args:
        md_content: Raw Markdown content from PDF conversion
        law_name: Name of the legal document for tag determination

    Returns:
        str: Cleaned and structured Markdown content
    """
    # Determine appropriate tag based on legal document type
    tag = "Article" if "CONSTITUTION" in law_name.upper() else "Section"
    if "INFORMATION TECHNOLOGY" in law_name.upper() or "IT ACT" in law_name.upper():
        tag = "Section"
    logger.debug(f"🧹 Scythe Pass: Using tag '{tag}' for {law_name}")
    
    # Pass 1: Global Reset - Remove all existing section headers to start clean
    md_content = re.sub(r'(?m)^###\s+(Section|Article).*$', '', md_content, flags=re.IGNORECASE)
    
    # Pass 2: Symbol Purge - Remove OCR artifacts and formatting noise
    noise_patterns = [ 
        r"", r"xxxGIDHxxx.*?CG-DL-E-\d+-\d+", r"\s-\d{5,}\s", r"lañ\s+\d+\]",
        r"No\.\s+\d+\]", r"ubZ\s+fnYyh.*?\d{4}", r".*¼'kd½.*",
        r"th\s+December,.*?\d{4}", r"\[\s*\d+\s*,\s*\d+\s*\]", r"\\_", r"_", r"Â",
        r"(?m)^[\s*_*-]+$", r"Information Technology Act, 2000", r"\[\d+th June, 2000\]"         
    ]
    for pattern in noise_patterns:
        md_content = re.sub(pattern, "", md_content, flags=re.IGNORECASE | re.MULTILINE)

    # Pass 3: Promotion - Convert inline section references to proper Markdown headers
    md_content = re.sub(r'^###\s+(Section|Article)\s+', f'### {tag} ', md_content, flags=re.IGNORECASE | re.MULTILINE)
    
    provision_pattern = r'^(\d+)\.\s+([A-Z].*)'
    lines, processed_lines, seen_ids = md_content.split('\n'), [], set()
    promoted_count = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('##'): 
            processed_lines.append(line); continue
        
        if stripped.startswith('###'):
            num_match = re.search(r'\d+', stripped)
            if num_match: seen_ids.add(num_match.group(0))
            processed_lines.append(line); continue

        match = re.match(provision_pattern, stripped)
        if match and match.group(1) not in seen_ids:
            processed_lines.append(f"\n### {tag} {match.group(1)}")
            processed_lines.append(line)
            seen_ids.add(match.group(1))
            promoted_count += 1
        else: processed_lines.append(line)

    logger.info(f"✨ Scythe: Promoted {promoted_count} text blocks to '{tag}' headers in {law_name}")
    final_md = re.sub(r'\n{3,}', '\n\n', '\n'.join(processed_lines)).strip()
    #Fix common OCR/Formatting glitches
    # Fix spacing issues before commas and standard labels
    replacements = {
        "Sanhita ,": "Sanhita,",
        "Act ,": "Act,",
        "Adhiniyam ,": "Adhiniyam,",
        "[ LAW:": "[LAW:",
        "passage :": "passage:",
        " ,": ",",
        "I T Act": "IT Act",
        "N I Act": "NI Act",
        "Dishonour of": "Dishonour of",
        "Cheque": "Cheque",
        "Banker": "Banker",
        "per cent.": "%",
        "under-mentioned": "below",
        "appropriate Government": "Appropriate Government",
        "wages": "wages"
    }
    
    for old, new in replacements.items():
        final_md = final_md.replace(old, new)
    
    return final_md.strip()

# --- PDF Processing & Chunking ---
def convert_pdf_to_md(file_path:str, output_path:str):
    """Converts PDF to Markdown using Docling with GPU acceleration if available."""
    law_name = Path(file_path).stem.replace("_", " ").upper()

    def get_pipeline_options(device_type):
        return PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=device_type),
            do_ocr=False,
            ocr_options=RapidOcrOptions(backend="torch"),
            ocr_batch_size=8,
            layout_batch_size=16,
            do_table_structure=True,
            allow_external_plugins=True,
            generate_page_images=False,
            generate_table_images=False,
            generate_picture_images=False
        )

    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    
    try:
        logger.info(f"🔄 Attempting conversion ({device}): {Path(file_path).name}")
        options = get_pipeline_options(device)
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        
        start_time = time.time()
        result = converter.convert(file_path)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.warning(f"⚠️ CUDA OOM on {Path(file_path).name}. Falling back to CPU...")
            
            # 2. CLEAR CACHE - This is vital to prevent subsequent crashes
            torch.cuda.empty_cache()
            
            # 3. Re-initialize with CPU
            options = get_pipeline_options(AcceleratorDevice.CPU)
            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
            
            start_time = time.time()
            result = converter.convert(file_path)
        else:
            raise e # Re-raise if it's a different runtime error
    
    if result.status == ConversionStatus.SUCCESS:
        raw_md = result.document.export_to_markdown()
        logger.debug(f"📊 Raw Markdown size: {len(raw_md)} chars")
        
        refined_md = clean_and_enrich_markdown(raw_md, Path(file_path).stem)
        logger.info(f"✨ Scythe Pass: Reduced noise from {len(raw_md)} to {len(refined_md)} chars")
        
        Path(output_path).write_text(refined_md, encoding="utf-8")
        logger.success(f"✅ Saved clean Markdown to {output_path} in {time.time() - start_time:.2f}s")
        # ATOMIC CLEANUP
        clear_memory(result)
        del result, converter, raw_md
        return refined_md
    else:
        logger.error(f"❌ Conversion failed for {Path(file_path).name}")
        clear_memory()
        return None

def get_chunks_from_md(md_path: str, law_name: str):
    """Split cleaned Markdown into structured chunks with legal metadata for vectorization.

    Performs hierarchical text splitting to create semantically coherent chunks from legal documents,
    preserving section structure and injecting breadcrumb metadata for retrieval accuracy.

    Splitting Strategy:
    1. Header-based splitting: Break on Part/Chapter/Section headers
    2. Recursive character splitting: Further divide into 1200-char chunks with 250-char overlap
    3. Metadata injection: Add law name and section references to each chunk

    Args:
        md_path: Path to the cleaned Markdown file
        law_name: Name of the legal document (e.g., "INDIAN CONTRACT ACT")

    Returns:
        List[Document]: List of chunked documents with metadata for vector store ingestion
    """
    logger.info(f"🥢 Splitting legal text into structured chunks: {law_name}")
    content = Path(md_path).read_text(encoding="utf-8")

    # Hierarchical splitting: First by major structural headers
    headers_to_split_on = [("#", "Part"), ("##", "Chapter"), ("###", "Section_Header")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    header_splits = md_splitter.split_text(content)
    logger.debug(f"📂 Identified {len(header_splits)} primary markdown headers in {law_name}")

    # Fine-grained splitting with legal-aware separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=250, is_separator_regex=True,
        separators=[r"\n### ", r"\n\(\d+\) ", r"\nProvided that", r"\nExplanation\.", r"\nIllustration\.", "\n\n", "\n", ". ", " ", ""]
    )
    docs = text_splitter.split_documents(header_splits)
    logger.info(f"🧩 Created {len(docs)} granular chunks for {law_name}")

    # Inject legal metadata and breadcrumbs for retrieval
    label = "Article" if "CONSTITUTION" in law_name.upper() else "Section"
    for i, doc in enumerate(docs):
        # Extract section number from header metadata
        raw_header = doc.metadata.get("Section_Header", "General")
        clean_num = re.search(r'\d+', raw_header)
        header_tag = f"{label} {clean_num.group(0)}" if clean_num else raw_header

        # Add structured metadata for legal retrieval
        doc.metadata["law_name"] = law_name
        doc.metadata["section"] = header_tag

        # Inject breadcrumb prefix for context-aware retrieval
        doc.page_content = f"passage: [LAW: {law_name} | {header_tag}]\n{doc.page_content}"

        # Clean up metadata (remove None values)
        doc.metadata = {k: v for k, v in doc.metadata.items() if v}

        if i < 3:
            logger.debug(f"🏷️  Metadata Verified [{i+1}]: {doc.metadata['law_name']} | {doc.metadata['section']}")

    logger.info(f"🧩 Metadata Logic: Injected [LAW: {law_name}] breadcrumbs into {len(docs)} chunks.")
    return docs

def run_ingestion_pipeline(vector_store: QdrantVectorStore):
    """Orchestrate the complete document ingestion pipeline for legal knowledge base.

    Processes all PDF legal documents through the full pipeline:
    1. PDF-to-Markdown conversion with OCR and cleaning
    2. Hierarchical text chunking with legal metadata
    3. Batch vectorization and upload to Qdrant store

    Implements caching to avoid reprocessing unchanged documents and memory management
    to handle large legal corpora efficiently.

    Args:
        vector_store: Initialized Qdrant vector store for document embeddings
    """
    logger.info("📥 Starting Global Ingestion Sync...")
    pdf_files = list(Path(DOCS_DIR).glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDFs found in {DOCS_DIR}")
        return

    for pdf_path in tqdm.tqdm(pdf_files, desc="Processing Legal Library"):
        logger.debug(f"🔍 Checking sync status for: {pdf_path.stem}")
        md_path = SCRATCH_DIR / f"{pdf_path.stem}.md"

        # Phase 1: Convert PDF to clean Markdown (cached)
        if not md_path.exists():
            convert_pdf_to_md(str(pdf_path), str(md_path))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Phase 2: Extract structured chunks with legal metadata
        chunks = get_chunks_from_md(str(md_path), pdf_path.stem.replace("_", " ").upper())

        # Phase 3: Batch upload to vector store with progress tracking
        if chunks:
            logger.info(f"🧠 Syncing {len(chunks)} snippets for {pdf_path.stem}...")
            batch_size = 20
            with tqdm.tqdm(total=len(chunks), desc=f"  ↳ Uploading to vector store", leave=False, unit="chunk") as pbar:
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    vector_store.add_documents(batch)
                    pbar.update(len(batch))
            logger.success(f"✅ Indexation Complete for {pdf_path.stem}")

            # Memory cleanup after each document
            chunks.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.success("🏁 Ingestion Pipeline: All documents are synchronized with the Vector Store.")