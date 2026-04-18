"""
Legal Document Retrieval Engine - Hybrid Vector Search System

This module implements a sophisticated retrieval system for legal documents using
hybrid vector search (dense + sparse embeddings) with advanced reranking and filtering.

Key Components:
- get_vector_store(): Qdrant vector database initialization with hybrid indexing
- get_retriever(): Multi-stage retrieval pipeline with query expansion and compression
- load_embeddings(): Cached HuggingFace embedding model loading
- get_reranker(): FlashRank model for result reranking

Features:
- Hybrid search combining semantic (dense) and keyword (sparse) retrieval
- Multi-query expansion for comprehensive search coverage
- Contextual compression to reduce token usage
- Payload filtering by legal act and section
- Dynamic thresholding based on legal domain
- GPU acceleration support with CUDA fallback
"""

from loguru import logger
from .prompts import mqr_prompt
from functools import lru_cache
from qdrant_client.http import models
from qdrant_client import QdrantClient
from .utils import StrictLegalQueryParser
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from .config import DEVICE, EMBED_MODEL_ID, CACHE_DIR, HF_TOKEN, QDRANT_URL, QDRANT_API_KEY

VECTOR_STORE = None  # Global vector store instance for caching

@lru_cache(maxsize=1)
def load_embeddings():
    """Load and cache the dense embedding model for semantic search.

    Uses HuggingFace's intfloat/e5-small-v2 model optimized for legal text understanding.
    Caches the heavy transformer model in memory to avoid reloads.

    Returns:
        HuggingFaceEmbeddings: Configured embedding model with GPU acceleration
    """
    logger.info("📡 Loading SentenceTransformer to RAM...")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={'device': DEVICE, 'token': HF_TOKEN},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32},
        cache_folder=str(CACHE_DIR / "huggingface")
    )

@lru_cache(maxsize=1)
def get_reranker():
    """Load and cache the FlashRank reranking model.

    Uses the ms-marco-MiniLM-L-12-v2 model for re-ranking search results
    by relevance to improve answer quality.

    Returns:
        FlashrankRerank: Configured reranking model
    """
    logger.info("🚀 Warming up FlashRank: ms-marco-MiniLM-L-12-v2")
    return FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=10)
    
def get_vector_store():
    """Initialize and cache the Qdrant hybrid vector store for legal documents.

    Creates a Qdrant collection with hybrid indexing (dense + sparse vectors) optimized
    for legal text retrieval. Includes payload indexing for metadata filtering.

    Features:
    - Dense embeddings: Semantic search using e5-small-v2
    - Sparse embeddings: Keyword search using FastEmbed BM25
    - Payload filters: Filter by law_name and section metadata
    - Hybrid retrieval: Combines semantic and keyword search
    - SSL security: Encrypted connection to Qdrant Cloud

    Returns:
        QdrantVectorStore: Configured hybrid vector store instance

    Raises:
        Exception: If vector store initialization fails
    """
    global VECTOR_STORE
    if VECTOR_STORE: return VECTOR_STORE

    logger.info("🛠️  Initializing Hybrid Vector Engine...")

    try:
        # Load the dense embedding model
        embeddings = load_embeddings()
        logger.debug(f"✅  Dense Embeddings loaded on {DEVICE}")

        # Initialize sparse embeddings for keyword search
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25", cache_dir=str(CACHE_DIR / "fastembed"))
        logger.debug("✅  Sparse BM25 Embeddings ready")

        # Connect to Qdrant Cloud with authentication
        # timeout=30 prevents indefinite hangs when Qdrant wakes from inactivity sleep
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        collection_name = "indian_legal_library"

        # Check if collection exists, create if not
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            logger.warning(f"⚠️  Collection '{collection_name}' not found. Creating schema...")
            # Create collection with hybrid vector configuration
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # Dense vectors
                sparse_vectors_config={"langchain-sparse": SparseVectorParams(index={"on_disk": True})}  # Sparse vectors
            )

            # Create payload indexes for efficient metadata filtering
            for field in ["metadata.law_name", "metadata.section"]:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

            logger.success(f"🆕  Schema created with Payload Indexing for '{collection_name}'")
        else:
            info = client.get_collection(collection_name)
            logger.info(f"💡  Collection '{collection_name}' active with {info.points_count} points")

        # Initialize LangChain Qdrant vector store with hybrid configuration
        VECTOR_STORE = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,  # Dense embeddings
            sparse_embedding=sparse_embeddings,  # Sparse embeddings
            sparse_vector_name="langchain-sparse",  # Sparse vector field name
            retrieval_mode=RetrievalMode.HYBRID  # Enable hybrid search
        )
        return VECTOR_STORE
    except Exception as e:
        logger.critical(f"🛑  Failed to initialize Vector Store: {e}")
        raise

def get_retriever(llm_for_queries, law_name_filter=None):
    """Create a multi-stage legal document retrieval pipeline.

    Builds a sophisticated retrieval system with query expansion, filtering,
    and reranking optimized for legal document search.

    Pipeline Stages:
    1. Multi-Query Expansion: Generate multiple search queries from user input
    2. Hybrid Retrieval: Combine dense (semantic) and sparse (keyword) search
    3. Payload Filtering: Filter results by legal act if specified
    4. Contextual Compression: Reduce token usage while preserving relevance
    5. FlashRank Reranking: Re-rank results by query relevance

    Args:
        llm_for_queries: Language model for generating expanded search queries
        law_name_filter: Optional filter for specific legal acts (e.g., "BNS", "CPC")

    Returns:
        ContextualCompressionRetriever: Configured retrieval pipeline
    """
    vs = get_vector_store()

    logger.debug("🔄  Configuring Multi-Query Expansion & FlashRank Reranker...")

    # Multi-query expansion prompt for comprehensive legal search
    MULTI_QUERY_PROMPT = mqr_prompt()

    # Base retrieval configuration with MMR (Maximal Marginal Relevance)
    search_kwargs={"k": 10, "fetch_k": 30, 'lambda_mult': 0.7}  # Retrieve 30, return 10 diverse results

    # Apply payload filtering if specific law requested
    if law_name_filter:
        logger.info(f"🎯 Applying Payload Filter: {law_name_filter}")
        if isinstance(law_name_filter, list):
            # Handle multiple law filters with OR logic
            match_logic = models.MatchAny(any=law_name_filter)
        else:
            # Single law filter
            match_logic = models.MatchValue(value=law_name_filter)
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.law_name",  # Filter by law name metadata
                    match=match_logic
                )
            ]
        )

    # Create base retriever with MMR for diversity
    base_ret = vs.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    # Add multi-query expansion for broader search coverage
    query_chain = MULTI_QUERY_PROMPT | llm_for_queries | StrictLegalQueryParser()
    mq_ret = MultiQueryRetriever(retriever=base_ret, llm_chain=query_chain, parser_key='lines')

    # Apply contextual compression to reduce token usage
    logger.debug("🚀  Loading FlashRank: ms-marco-MiniLM-L-12-v2")
    compressor = get_reranker()

    logger.success("🚀  Retrieval Engine: Multi-Query + FlashRank Reranker is ready.")
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=mq_ret)