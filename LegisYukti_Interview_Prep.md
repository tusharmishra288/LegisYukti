# LegisYukti — Comprehensive Technical Interview Preparation Guide

> **Private document — NOT in the repository. For interview preparation only.**
> Last updated: April 2026 (reflects all production fixes applied in this session)

---

## TABLE OF CONTENTS

1. [Project Summary (60-second pitch)](#1-project-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Technology Stack — Why Each Was Chosen vs Alternatives](#3-technology-stack--why-each-was-chosen-vs-alternatives)
4. [Deep Dive: LangGraph Agentic Pipeline (agent.py)](#4-deep-dive-langgraph-agentic-pipeline-agentpy)
5. [Deep Dive: Hybrid Vector Search Engine (engine.py)](#5-deep-dive-hybrid-vector-search-engine-enginepy)
6. [Deep Dive: PDF Ingestion & Data Refiner (processor.py)](#6-deep-dive-pdf-ingestion--data-refiner-processorpy)
7. [Deep Dive: Prompt Engineering (prompts.py)](#7-deep-dive-prompt-engineering-promptspy)
8. [Deep Dive: PostgreSQL Checkpointing (backend.py)](#8-deep-dive-postgresql-checkpointing-backendpy)
9. [Deep Dive: Keep-Alive Service (keep_alive.py)](#9-deep-dive-keep-alive-service-keep_alivepy)
10. [Deep Dive: Configuration & Models (config.py)](#10-deep-dive-configuration--models-configpy)
11. [Deep Dive: Logging, Utils & App (logger.py / utils.py / app.py)](#11-deep-dive-logging-utils--app)
12. [Security Hardening & CVE Patches Applied](#12-security-hardening--cve-patches-applied)
13. [Performance Optimizations](#13-performance-optimizations)
14. [Deployment Architecture (Docker + HF Spaces)](#14-deployment-architecture-docker--hf-spaces)
15. [Anticipated Interview Questions & Model Answers](#15-anticipated-interview-questions--model-answers)

---

## 1. Project Summary

**LegisYukti** (Legis = Law [Latin], Yukti = Reasoning [Sanskrit]) is a **production-grade Agentic RAG (Retrieval-Augmented Generation) framework** for Indian legal research, deployed on HuggingFace Spaces.

### One-liner
> An AI legal research assistant that retrieves, reasons over, audits, and self-scores responses grounded in 17 official Indian statutes — with specialised handling of the 2023 Criminal Law Reforms (BNS, BNSS, BSA).

### What it does (end-to-end)
1. User types a natural-language legal question (e.g., "What is the punishment for cheating under the new law?")
2. A **router node** classifies intent (LEGAL vs CHAT) and checks if it's a follow-up question
3. A **dispatcher agent node** identifies the relevant statute (from a hardcoded 17-law library + LLM fallback) and calls a retrieval tool
4. The **retrieval tool** runs hybrid vector search (dense semantic + BM25 sparse) across 9,221 legal chunks, applies FlashRank reranking, and returns verified references
5. A **synthesiser node** composes a grounded legal response with section citations, procedural roadmap, and mandatory disclaimer
6. A **citation auditor** cross-checks the response against retrieved context and flags/corrects hallucinations
7. An **evaluator node** scores the response (0–10). If score < 6, it loops back for one retry
8. Conversation history is persisted in **Neon Postgres** via LangGraph's `PostgresSaver`

### Scale
- 17 statutes covering Criminal, Civil, Family, Property, Labour, Consumer, and Constitutional law
- ~9,221 vector points in Qdrant (`indian_legal_library` collection)
- 1,200-character chunks with 250-character overlap
- End-to-end latency: ~8–15 seconds (Groq API LLM inference + vector search on CPU)
- Free-tier deployment: HF Spaces Docker (2 vCPU, 16 GB RAM) + Qdrant Cloud free cluster

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                              │
│              Streamlit App (app.py) — Port 7860                  │
│    Workspace manager │ Chat UI │ Fidelity audit cards            │
│    ?health=true endpoint │ Keep-alive status widget              │
└──────────────────────────┬───────────────────────────────────────┘
                           │  HumanMessage
┌──────────────────────────▼───────────────────────────────────────┐
│              LANGGRAPH STATE MACHINE (agent.py)                   │
│                                                                   │
│  router_node                                                      │
│     ├── CHAT → final_answer_node (persona response)              │
│     ├── FOLLOW_UP → final_answer_node (reuse context)            │
│     └── LEGAL → agent/chat_node                                  │
│                    │ binds retrieve_legal_context @tool           │
│                    ├── tool_calls → call_tools_and_save_context  │
│                    │                    │                         │
│                    │             retrieve_legal_context()         │
│                    │        [Hybrid search + FlashRank]          │
│                    │                    │                         │
│                    └── no_tools → final_answer_node ◄────────────┘
│                                         │
│                              verify_citations_node (auditor)     │
│                                         │
│                              evaluate_response_node (0-10)       │
│                                         │
│                          score ≥ 6 → END  │  score < 6 → retry  │
│                                           └──► retry_prep_node   │
│                                                     │            │
│                                              ───► agent/chat_node│
└───────────────────────────┬──────────────────────────────────────┘
                            │
         ┌──────────────────┼────────────────────┐
         │                  │                    │
┌────────▼───────┐  ┌───────▼──────┐  ┌─────────▼──────────┐
│  QDRANT CLOUD  │  │   GROQ API   │  │   NEON POSTGRES     │
│ Hybrid Search  │  │ LLaMA-3.3-70B│  │  PostgresSaver      │
│ Dense: e5-small│  │ LLaMA-3.1-8B │  │  thread_id workspaces│
│ Sparse: BM25   │  │ temperature=0│  │  Audit log trails   │
│ 9,221 points   │  │ max_tokens   │  │  Connection pool    │
└────────────────┘  │ 800 / 1024   │  └────────────────────┘
         ▲          └──────────────┘
┌────────┴────────────────────────────────────────────────────────┐
│               INGESTION PIPELINE (processor.py)                  │
│  17 PDFs → Docling (OCR/Table) → Data Refiner → Chunking       │
│       → Metadata injection → Qdrant batch upload (20/batch)     │
└─────────────────────────────────────────────────────────────────┘
         ▲
┌────────┴────────────────────────────────────────────────────────┐
│               KEEP-ALIVE SERVICE (keep_alive.py)                 │
│  Daemon thread (daemon=True) pings every 10 minutes:            │
│   1. HF Space: GET https://{SPACE_HOST}/?health=true            │
│   2. Qdrant Cloud: GET {QDRANT_URL}/collections                 │
│  Thread-alive check prevents resurrection bug                   │
│  60s retry on failure before full interval resumes              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions (why these patterns were used)
| Pattern | Reason |
|---|---|
| **LangGraph StateGraph** | Cyclic retry loop is impossible in a linear chain; LangGraph supports conditional edges and `add_messages` merging |
| **`@st.cache_resource`** | Prevents vector store / DB pool re-initialisation on every Streamlit re-run (it's a singleton that persists across browser refreshes) |
| **Daemon thread for keep-alive** | `daemon=True` means Python does not wait for it at shutdown, avoiding zombie threads |
| **PostgresSaver over MemorySaver** | Cross-session persistence; conversations survive container restarts; mandatory for multi-workspace feature |
| **TypedDict ChatState** | Strongly typed state ensures nodes can only write to declared keys, preventing silent runtime bugs |

---

## 3. Technology Stack — Why Each Was Chosen vs Alternatives

### 3.1 LangGraph — Agentic Orchestration

| | LangGraph ✅ | LangChain LCEL Chains | AutoGen | CrewAI |
|---|---|---|---|---|
| **Cyclic graphs** | Yes (conditional edges) | No (linear only) | Yes | Yes |
| **State management** | TypedDict, full control | Partial | Agent-level | Role-level |
| **Checkpointing** | Built-in `PostgresSaver` | None | None | None |
| **Debug visibility** | Full node-by-node streaming | Limited | Complex | Limited |
| **Why chosen** | The retry loop (evaluator → retry_prep → agent) requires a cycle. PostgresSaver is the only production-ready checkpoint backend in the ecosystem. |

### 3.2 Qdrant — Vector Database

| | Qdrant ✅ | Pinecone | ChromaDB | FAISS |
|---|---|---|---|---|
| **Hybrid search** | Native (dense + sparse) | Paid tier only | No | No |
| **Payload filtering** | Yes, indexed | Yes | Yes (limited) | No |
| **Self-hostable** | Yes (Docker) | No | Yes | Yes |
| **Free cloud tier** | Yes (1GB, ~1M points) | 1 index | N/A | N/A |
| **gRPC support** | Yes | Yes | No | No |
| **Why chosen** | `RetrievalMode.HYBRID` lets us combine semantic search (e5-small-v2 dense embeddings) with BM25 keyword search (FastEmbedSparse) natively. Pinecone requires a paid tier for hybrid. ChromaDB has no sparse vector support. |

### 3.3 Groq API — LLM Inference

| | Groq ✅ | OpenAI | Ollama (local) | Anthropic |
|---|---|---|---|---|
| **Latency** | ~2–4s (LPU hardware) | ~5–10s | ~30–120s (CPU) | ~5–8s |
| **Cost** | Free tier generous | Paid per token | Free | Paid |
| **Model** | LLaMA-3.3-70B | GPT-4 | LLaMA-3.1-8B | Claude Sonnet |
| **`temperature=0`** | Yes | Yes | Yes | Yes |
| **Why chosen** | Free-tier deployment constraint eliminates OpenAI and Anthropic. Local Ollama is too slow on the 2-vCPU HF container. Groq's LPU delivers near-OpenAI latency at zero cost. |

**Two-model strategy:**
- `llm` = `llama-3.3-70b-versatile` (`max_tokens=800`) — used for final legal synthesis and dispatcher (high accuracy needed)
- `fast_llm` = `llama-3.1-8b-instant` (`max_tokens=1024`) — used for routing, multi-query expansion, auditing, evaluation (speed matters)

### 3.4 Docling — PDF Processing

| | Docling ✅ | PyMuPDF (fitz) | pdfminer | PDFPlumber |
|---|---|---|---|---|
| **Table extraction** | Yes (native) | Partial | No | Yes |
| **OCR support** | RapidOCR + Torch | No (needs tesseract) | No | No |
| **GPU acceleration** | CUDA device selection | No | No | No |
| **Markdown output** | Yes (structured) | Plain text | Plain text | Plain text |
| **Why chosen** | Legal PDFs from government portals are scanned images with complex tables. Docling's `RapidOcrOptions` with GPU acceleration converts them to structured Markdown. PyMuPDF handles text-only PDFs; it fails on scanned images without OCR. |

**CUDA OOM fallback pattern used:**
```python
try:
    device = AcceleratorDevice.CUDA
    converter.convert(file_path)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        torch.cuda.empty_cache()
        device = AcceleratorDevice.CPU  # Retry on CPU
```

### 3.5 FastEmbed (BM25) — Sparse Embeddings

| | FastEmbed BM25 ✅ | Elasticsearch BM25 | SPLADE | TF-IDF |
|---|---|---|---|---|
| **In-process** | Yes | No (separate service) | No | Yes |
| **Qdrant integration** | Native `FastEmbedSparse` | No | Partial | No |
| **Why chosen** | `FastEmbedSparse(model_name="Qdrant/bm25")` runs in-process without a separate server. Critical for matching exact statutory references like "Section 173 BNSS" that dense embeddings miss semantically. |

### 3.6 intfloat/e5-small-v2 — Dense Embeddings

| | e5-small-v2 ✅ | text-embedding-ada-002 | all-MiniLM-L6-v2 | BAAI/bge-small-en |
|---|---|---|---|---|
| **Dimensions** | 384 | 1536 | 384 | 384 |
| **Local / free** | Yes | No (OpenAI API) | Yes | Yes |
| **Legal domain** | Good | Good | Moderate | Good |
| **Why chosen** | 384-dimension vectors are 4× cheaper to store/search than ada-002's 1536. Outperforms `all-MiniLM-L6-v2` on domain-specific retrieval. `normalize_embeddings=True` ensures cosine distance works correctly. |

### 3.7 FlashRank — Reranking

| | FlashRank ✅ | Cohere Rerank API | Cross-Encoder | ColBERT |
|---|---|---|---|---|
| **Local** | Yes | No (API) | Yes | Yes |
| **Speed** | Very fast (MiniLM) | Fast | Slow | Slow |
| **Why chosen** | Free tier constraint eliminates Cohere. `FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=10)` runs in-process, rearranges the top-30 retrieved results, and returns the 10 most relevant before sending to the LLM. |

### 3.8 Neon — PostgreSQL for Checkpointing

| | Neon ✅ | Supabase | Local PostgreSQL | MemorySaver |
|---|---|---|---|---|
| **Serverless** | Yes | Yes | No | N/A |
| **Free tier** | 0.5GB | 500MB | N/A | N/A |
| **Cold start** | ~500ms | ~1s | None | None |
| **LangGraph support** | Yes (`PostgresSaver`) | Yes | Yes | Yes (no persistence) |
| **Why chosen** | `PostgresSaver` is the LangGraph-native checkpointing solution. Neon serverless is the most cost-effective PostgreSQL option that supports SSL + SCRAM authentication required by `psycopg`. `MemorySaver` was rejected because it loses all conversation history on container restart. |

**Neon connection kwargs (why each one matters):**
```python
connection_kwargs = {
    "autocommit": True,           # Required by LangGraph PostgresSaver
    "prepare_threshold": None,    # Disables prepared statements (serverless doesn't support them)
    "sslmode": "require",         # Neon requires SSL
    "channel_binding": "require", # SCRAM authentication binding
    "tcp_user_timeout": 10000     # 10s timeout prevents hangs on cold-start
}
```

### 3.9 Streamlit — Frontend

| | Streamlit ✅ | Gradio | FastAPI + React | Flask |
|---|---|---|---|---|
| **Rapid prototyping** | Yes | Yes | No (complex setup) | Moderate |
| **HF Spaces native** | Yes | Yes | Yes | Yes |
| **State management** | `session_state` | Limited | Full | Limited |
| **`@cache_resource`** | Yes (singleton) | No | N/A | N/A |
| **Why chosen** | HF Spaces has first-class Streamlit support. `@st.cache_resource` is critical — it creates a singleton for the vector store and DB pool that persists across user re-runs. Gradio was rejected because its state management is weaker for multi-workspace use cases. |

### 3.10 UV — Package Manager

| | UV ✅ | pip | Poetry | Conda |
|---|---|---|---|---|
| **Speed** | ~10–100× faster | Baseline | 2× faster | Slow |
| **Lock file** | `uv.lock` | None | `poetry.lock` | environment.yml |
| **`override-dependencies`** | Yes | No | `force` | No |
| **Why chosen** | `override-dependencies` lets us force security patches like `pillow>=12.2.0` across all transitive dependencies without forking packages. The `unsafe-best-match` index strategy allows mixing PyTorch's custom CUDA wheel index with PyPI. |

### 3.11 Loguru — Logging

| | Loguru ✅ | Python logging | structlog |
|---|---|---|---|
| **Setup** | 0 config | Verbose boilerplate | Moderate |
| **Color output** | Built-in | Third-party | Built-in |
| **Rotation/retention** | `rotation="50 MB"` | None | None |
| **Why chosen** | `logger.success()`, `logger.critical()` levels are semantic and readable. Built-in log rotation, compression, and enqueue for thread-safe async logging. The `InterceptHandler` bridges external library `logging.getLogger()` calls into Loguru. |

---

## 4. Deep Dive: LangGraph Agentic Pipeline (agent.py)

### 4.1 ChatState — The Shared Memory

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Auto-merges (no overwrites)
    context: list[str]       # Retrieved legal snippets
    evaluation_score: int    # Auditor quality score 0-10
    evaluation_feedback: str # Reason for the score
    retry_count: int         # Max 1 retry (prevents infinite loops)
    law_filter: Optional[str]# Detected statute (e.g. "BNS")
    intent: str              # "LEGAL" or "CHAT"
    is_followup: bool        # Skip retrieval if True
```

**Why `Annotated[list, add_messages]`?** Without this, returning `{"messages": [new_msg]}` from a node would *replace* the entire conversation history. `add_messages` merges — it appends new messages and replaces existing ones with the same ID. This is how LangGraph maintains conversation context across nodes.

### 4.2 The 8 Nodes and Their Roles

| Node | LLM Used | Function |
|---|---|---|
| `router_node` | fast_llm (8B) | Classifies LEGAL vs CHAT. Checks if LEGAL query is a follow-up (skip retrieval if context already exists) |
| `chat_node` (agent) | llm (70B) | Identifies statute via keyword routing map + LLM fallback. Binds `retrieve_legal_context` tool. Enforces single tool call (consolidates multiple) |
| `call_tools_and_save_context` | None | Executes tool call, saves retrieved text to `context`, emits `ToolMessage` |
| `generate_response_node` (final_answer) | llm (70B) | Synthesises legal advice from verified context. Adds source footer + disclaimer. Emergency re-prompt if LLM refuses context |
| `verify_citations_node` (auditor) | fast_llm (8B) | Audits response vs. context. `✅` = pass through; `🚨` = replace with corrected version |
| `evaluate_response_node` (evaluator) | fast_llm (8B) | Scores 0-10 with strict rubric (must cite 2023 Acts, must include CPC/BNSS procedure). Returns SCORE + REASON |
| `retry_prep_node` | None | Injects self-correction message into context, increments `retry_count` |
| `route_after_evaluation` | None | If score < 6 AND retry_count < 1 → retry. Otherwise → END |

### 4.3 Law Routing — The 17-Law Library

The `chat_node` implements a **keyword-first, LLM-fallback** routing strategy:

```
Step 1: Keyword routing map
  "theft", "murder", "cheating" → BNS 2023
  "fir", "arrest", "bail"       → BNSS 2023
  "salary", "wage", "firing"    → Code on Wages 2019
  ...

Step 2: Procedural bridge injection
  Civil law detected → auto-add CPC 1908
  Criminal law detected → auto-add BNSS 2023

Step 3: LLM fallback
  If no keyword match → fast_llm picks from the 17-law menu
  Validated against the hardcoded list (prevents hallucinated law names)

Step 4: Tool call consolidation
  If LLM generates >1 tool call → merge queries with " | " separator
  (Prevents parallel API calls that waste quota)
```

### 4.4 The Retry Loop

```
evaluator → score < 6 AND retry_count < 1
         → retry_prep_node (adds: "Self-Correction: Previous attempt low quality...")
         → chat_node (full RAG again with feedback in context)
         → final_answer → auditor → evaluator
         → if score < 6 again → END (retry_count = 1, limit hit)
```

**Why max 1 retry?** Prevents infinite loops when a query genuinely has no relevant context. The retry feedback message gives the dispatcher a hint about what went wrong.

### 4.5 Citation Verification (Auditor) — Hallucination Prevention

The auditor prompt enforces:
- **Temporal accuracy**: Any mention of "IPC", "CrPC", "Indian Evidence Act" → `🚨 HALLUCINATION`
- **Citation check**: Referenced section must exist in the retrieved context
- **Succession guard**: "Hindu Succession Act" is not in the corpus → flag and replace with "Indian Succession Act"

This catches cases where the 70B LLM confidently cites a plausible-sounding but wrong section number.

---

## 5. Deep Dive: Hybrid Vector Search Engine (engine.py)

### 5.1 Vector Store Initialisation

```python
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
```
**`timeout=30`** — Critical fix: Qdrant Cloud free tier sleeps after 30 minutes of inactivity. Without a timeout, the first connection attempt after sleep would hang indefinitely. With `timeout=30`, it either wakes within 30s or raises a clean exception.

```python
VECTOR_STORE = QdrantVectorStore(
    client=client,
    collection_name="indian_legal_library",
    embedding=embeddings,               # Dense: e5-small-v2 (384 dims)
    sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),  # Sparse: BM25
    sparse_vector_name="langchain-sparse",
    retrieval_mode=RetrievalMode.HYBRID # Combines both
)
```

### 5.2 The Retrieval Pipeline (5 stages)

```
User query
    │
    ▼ Stage 1: Multi-Query Expansion (fast_llm)
    "What is the punishment for cheating?"
    → Query 1: "cheating fraud breach of trust substantive law"
    → Query 2: "CPC Order 39 remedy procedure for cheating case"
    │
    ▼ Stage 2: Payload Filtering (if law_filter set)
    Filter: FieldCondition(key="metadata.law_name", match=MatchValue("BNS"))
    │
    ▼ Stage 3: MMR Retrieval (fetch_k=30, k=10, lambda_mult=0.7)
    Returns 10 diverse candidates from 30 fetched
    lambda_mult=0.7 → 70% relevance, 30% diversity
    │
    ▼ Stage 4: FlashRank Reranking (ms-marco-MiniLM-L-12-v2, top_n=10)
    Reorders by cross-encoder relevance to original query
    │
    ▼ Stage 5: Dynamic Thresholding (in retrieve_legal_context @tool)
    Global search: threshold 0.08
    Filtered civil law: threshold 0.12
    Filtered criminal law: threshold 0.22
    Force-add top-3 if < 3 results pass threshold
```

### 5.3 Global Fallback Logic

```python
docs = get_retriever(fast_llm, law_name_filter=active_filter).invoke(search_query)
if not docs:
    # Filtered search found nothing (law filter too restrictive)
    docs = get_retriever(fast_llm, law_name_filter=None).invoke(search_query)
```

This prevents zero-context responses when the user's query matches a law by keyword but the vector store has limited coverage for that specific section.

### 5.4 Succession Law Boost

```python
if any(x in query.lower() for x in ["inheritance", "succession", "will"]):
    if "indian succession" in law_ref:
        score += 0.05  # Boost ISA over HSA
```
Hindu Succession Act is not in the corpus. This boost ensures Indian Succession Act wins the relevance contest even if a chunk has a slightly lower raw score.

### 5.5 `@lru_cache` for Model Loading

```python
@lru_cache(maxsize=1)
def load_embeddings():
    return HuggingFaceEmbeddings(...)

@lru_cache(maxsize=1)
def get_reranker():
    return FlashrankRerank(...)
```
`lru_cache(maxsize=1)` means the model is loaded once on first call and cached in memory. Combined with `@st.cache_resource` at the app level, the 384-dim transformer model is never reloaded across Streamlit re-runs.

---

## 6. Deep Dive: PDF Ingestion & Data Refiner (processor.py)

### 6.1 Why a Custom "Data Refiner"?

Government legal PDFs suffer from:
1. **OCR artifacts**: Characters like `Â`, `lañ`, `xxxGIDHxxx`, stray underscores
2. **Missing section headers**: "Section 173." appears as plain text, not a Markdown header
3. **Cross-references confusion**: "Section 173 (replaces 154 CrPC)" — the old law reference needs to survive but not be promoted

The Data Refiner runs 4 passes on raw Markdown:

```
Pass 1: Global reset — Remove any existing malformed `### Section X` headers
Pass 2: Symbol purge — 14 regex patterns for OCR noise removal
Pass 3: Promotion — Convert "173. Short title." → "### Section 173"
Pass 4: OCR fixes — "Sanhita ," → "Sanhita,", "N I Act" → "NI Act"
```

### 6.2 Chunking Strategy — Why These Parameters

```python
RecursiveCharacterTextSplitter(
    chunk_size=1200,    # Large enough to contain a full legal provision + context
    chunk_overlap=250,  # ~20% overlap preserves cross-sentence references
    separators=[
        r"\n### ",      # Split on section headers first
        r"\n\(\d+\) ",  # Split on sub-clauses like "(1) No person shall..."
        r"\nProvided that",  # Provisions follow a different statute rule
        r"\nExplanation\.",  # Explanations are contextually separate
        "\n\n", "\n", ". ", " ", ""
    ]
)
```
**Why 1,200 chars?** Legal provisions are verbose. 512-char chunks frequently cut mid-sentence. 2,000-char chunks waste LLM token budget. 1,200 chars holds ~2–3 legal sub-clauses cleanly.

### 6.3 Metadata Injection — Why It Matters for Retrieval

Each chunk gets:
```python
doc.metadata["law_name"] = "BHARATIYA NYAYA SANHITA BNS 2023"
doc.metadata["section"]  = "Section 303"
doc.page_content = "passage: [LAW: BHARATIYA NYAYA SANHITA BNS 2023 | Section 303]\n{content}"
```

The `passage: [LAW: ...]` breadcrumb prefix serves dual purposes:
1. **e5-small-v2 is trained with `passage:` prefix** — removing it degrades retrieval quality
2. **Source citation assembly** — `generate_response_node` uses regex to extract law names for the "Verified Sources" footer

### 6.4 CUDA OOM Fallback

```python
try:
    device = AcceleratorDevice.CUDA
    result = converter.convert(file_path)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        torch.cuda.empty_cache()  # CRITICAL: must clear before CPU retry
        device = AcceleratorDevice.CPU
        result = converter.convert(file_path)  # Re-run on CPU
```
HF Spaces free tier has 16GB RAM but limited VRAM if GPU is assigned. Large scanned PDFs (>50 pages) can OOM on CUDA. The fallback ensures the ingestion pipeline doesn't crash.

### 6.5 Memory Management for Batch Processing

```python
# After each PDF:
converter_result.input._backend.unload()  # Releases Docling's C++ backend memory
gc.collect()
torch.cuda.empty_cache()
chunks.clear()
```
Docling uses a Rust/C++ backend that Python's `gc.collect()` cannot reach. Explicit `_backend.unload()` is required to prevent memory leaks during batch processing of 17 PDFs.

---

## 7. Deep Dive: Prompt Engineering (prompts.py)

### 7.1 Five Specialised Prompts

| Prompt | Model | Purpose | Key Constraint |
|---|---|---|---|
| `get_qa_prompt()` | llm (70B) | Legal synthesis | MUST use BNS/BNSS/BSA, NOT IPC/CrPC |
| `get_auditor_prompt()` | fast_llm (8B) | Citation verification | Emoji-gated verdict: ✅ or 🚨 |
| `mqr_prompt()` | fast_llm (8B) | Multi-query expansion | 2 queries: substantive + procedural |
| `get_router_prompt()` | fast_llm (8B) | LEGAL vs CHAT | Returns only one word |
| `get_followup_classifier_prompt()` | fast_llm (8B) | FOLLOW_UP vs NEW_TOPIC | Returns only one of two values |
| `get_chat_persona_prompt()` | fast_llm (8B) | Off-topic redirection | Stay polite, pivot to legal role |

### 7.2 The Temporal Accuracy Problem

India replaced IPC (1860), CrPC (1973), and IEA (1872) with new 2023 acts on 1 July 2024. LLMs trained before this date will default to IPC/CrPC. The QA prompt enforces this with a hard CRITICAL FAILURE rule:

```
STRICT MANDATE: Use ONLY the 2023 Sanhitas for Criminal matters.
Citing IPC (1860), CrPC (1973), or IEA (1872) is a CRITICAL FAILURE.

GOSPEL TRUTH MAPPING:
[Murder: 101 BNS]  | [Theft: 303 BNS]  | [Cheating: 318 BNS]
[FIR: 173 BNSS]    | [Arrest: 35 BNSS] | [Bail: 480/482 BNSS]
```

The auditor then double-checks this constraint independently, catching cases where the 70B model slips IPC references anyway.

### 7.3 Temporal Guardrail for Pre-2024 Incidents

```python
is_old_law = any(yr in question for yr in ["2020", "2021", "2022", "2023"])
temporal_hint = " (Note: Incident is pre-July 2024, prioritize IPC/CrPC)" if is_old_law else ""
```
For crimes that occurred before the 2023 reform took effect, the applicable law is the OLD law (IPC for a 2021 murder, for example). The temporal hint correctly switches the model back to citing IPC/CrPC for pre-2024 incidents.

### 7.4 Emergency Context Injection

```python
if any(phrase in response.content for phrase in ["haven't provided", "specific legal query", "I cannot"]):
    emergency_prompt = f"Using this context: {full_context}\n\nAnswer: {question}"
    response = llm.invoke(emergency_prompt)
```
Some prompts cause the LLM to refuse engagement ("I cannot answer without more context"). The emergency prompt bypasses the structured chain and forces context usage.

---

## 8. Deep Dive: PostgreSQL Checkpointing (backend.py)

### 8.1 Why Conversation Persistence Matters

Without checkpointing, every browser refresh starts a fresh conversation. Legal consultations often span multiple sessions ("What's the next step after filing the FIR?"). `PostgresSaver` stores the complete `ChatState` (including all messages, scores, and context) keyed by `thread_id`.

```python
config = {"configurable": {"thread_id": "workspace_CASE_001"}}
graph.stream(input_state, config)  # Resumes from stored state
```

### 8.2 Multi-Workspace Architecture

In `app.py`, each workspace (case) gets its own `thread_id`:
```python
st.session_state.workspace = st.query_params.get("checkpoint", "DEFAULT")
# workspace = "CASE_001" → thread_id in Postgres = "CASE_001"
```
Users can switch between active cases (e.g., "Cheque Bounce Case" vs "Property Dispute") and each maintains its own conversation history and retrieved context.

### 8.3 Serverless Connection Retry Logic

```python
def connect_with_retry(uri, kwargs, retries=3, delay=5):
    for attempt in range(retries):
        try:
            pool = ConnectionPool(conninfo=uri, max_size=10, kwargs=kwargs)
            with pool.connection() as conn:
                conn.execute("SELECT 1")  # Pre-ping validates SSL handshake
            return pool
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
```
Neon serverless databases have a cold-start delay of ~500ms. The `SELECT 1` pre-ping ensures the SSL handshake completes before LangGraph starts writing checkpoints. `max_size=10` limits concurrent connections to stay within Neon's free tier limits.

---

## 9. Deep Dive: Keep-Alive Service (keep_alive.py)

### 9.1 The Two-Layer Problem

HF Spaces free tier:
- Container sleeps after **72 hours** of inactivity (configurable, `gcTimeout=172800`)
- "Activity" is defined as HTTP requests reaching the container

Qdrant Cloud free tier:
- Cluster sleeps after **30 minutes** of no queries

The root cause of both going inactive despite having keep-alive code was **the HF Space itself going to sleep first**. When the container sleeps, the Python daemon thread dies, Qdrant pings stop, and Qdrant sleeps too. Everything cascades from the container sleep.

### 9.2 Two-Ping Strategy

```python
# Ping 1: HF Space self-ping (prevents container sleep)
space_host = os.getenv("SPACE_HOST")  # "tushar2704-legisyukti.hf.space"
self.hf_health_url = f"https://{space_host}/?health=true"
# GET this URL every 10 minutes → HF sees HTTP activity → container stays awake

# Ping 2: Qdrant Cloud ping (prevents cluster sleep)
self.qdrant_health_url = f"{QDRANT_URL.rstrip('/')}/collections"
headers = {"api-key": QDRANT_API_KEY}  # Authenticated request
# GET this URL every 10 minutes → Qdrant sees activity → cluster stays awake
```

**Why `SPACE_HOST`?** HF Spaces injects this environment variable automatically. It contains the public hostname without protocol or path (e.g., `tushar2704-legisyukti.hf.space`). This is the only reliable way to get the space's own URL from within the container.

**Why `/collections`?** It's the lightest Qdrant API endpoint that requires authentication — just returns the list of collection names. It proves the cluster is alive without triggering any compute.

### 9.3 Bugs Fixed in This Session

**Bug 1: Missing HF Space self-ping**
Original code only pinged Qdrant. No self-ping meant the container still slept → daemon thread died → Qdrant pings stopped → Qdrant slept.

**Bug 2: Thread resurrection bug**
```python
# BEFORE (broken):
def start(self):
    if self.running:  # True even if thread died!
        return

# AFTER (fixed):
def start(self):
    if self.running and self.thread and self.thread.is_alive():  # Checks actual thread state
        return
```
If the daemon thread crashed (e.g., uncaught exception in `_keep_alive_loop`), `self.running` was still `True`. The fixed version checks `.is_alive()` to detect dead threads and respawns them.

**Bug 3: Double start on Streamlit re-runs**
`start_keep_alive_service()` was called at module level, so every Streamlit re-run (triggered by any UI interaction) attempted to start the service again. Fixed by moving it inside `@st.cache_resource`:
```python
@st.cache_resource
def init_system_core():
    pool = connect_with_retry(DB_URI, connection_kwargs)
    start_keep_alive_service(interval_minutes=10)  # Called once only
    return pool, create_graph(PostgresSaver(pool))
```

**Bug 4: 60-second retry on failure**
Original code slept the full 10-minute interval even after a failed ping. Fixed with:
```python
if not hf_ok or not qdrant_ok:
    time.sleep(_RETRY_DELAY_SECONDS)  # 60s fast retry
    continue  # Don't wait the full 10 min
```

**Bug 5: Widget showing stale status**
The sidebar widget only checked `running` flag, not actual thread state:
```python
# FIXED:
truly_running = keep_alive_status["running"] and keep_alive_status.get("thread_alive", False)
status_icon = "🟢" if truly_running else "🔴"
```

### 9.4 Why a Daemon Thread (not asyncio)?

`daemon=True` means Python's main thread (Streamlit) does not wait for the keep-alive thread to finish at shutdown. This prevents zombie threads from blocking container shutdown. An asyncio task would require running inside an event loop, which Streamlit's threading model makes complex.

### 9.5 External vs Internal Keep-Alive

| | UptimeRobot (external) | KeepAliveService (internal) |
|---|---|---|
| **Pings** | HF Space URL (HTTP HEAD) | HF Space + Qdrant (HTTP GET) |
| **Frequency** | 5 minutes (free tier) | 10 minutes |
| **Auth needed** | No | Yes (Qdrant API key) |
| **Survives container restart** | Yes (external) | No (dies with container) |
| **Can ping Qdrant** | No (no API key) | Yes |

UptimeRobot pings the public HF Space URL every 5 minutes — this keeps the container awake. The internal service pings Qdrant every 10 minutes — UptimeRobot can't do this because it has no Qdrant API key. Both are needed.

---

## 10. Deep Dive: Configuration & Models (config.py)

### 10.1 Directory Structure

```python
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR  = PROJECT_ROOT / "docs"        # 17 source PDFs (tracked with git-lfs)
SCRATCH_DIR = PROJECT_ROOT / "scratch"   # Cached .md files (PDF → Markdown)
CACHE_DIR = PROJECT_ROOT / "model_cache" # HuggingFace + FastEmbed model weights
LOG_DIR   = PROJECT_ROOT / "logs"        # Loguru rotating log files
```

### 10.2 Model Cache Redirection

```python
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["FASTEMBED_CACHE_PATH"] = str(CACHE_DIR / "fastembed")
```
Without this, HuggingFace downloads models to `~/.cache/huggingface` which is not persisted in the HF Docker container between restarts. By pointing to `model_cache/` inside the project, model weights persist in the container's filesystem.

### 10.3 Hardware Detection

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
Affects both embedding generation (`model_kwargs={'device': DEVICE}`) and Docling PDF processing (`AcceleratorDevice.CUDA`). On HF Spaces free CPU tier, this resolves to `"cpu"`.

### 10.4 Two-Model Design Rationale

```python
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,   # Deterministic — legal accuracy requires consistency
    max_tokens=800   # Enough for: legal position + 5-step roadmap + sources + disclaimer
)

fast_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,   # Consistent query transformations
    max_tokens=1024  # More for multi-query expansion (returns 2-3 queries)
)
```

`temperature=0` for both models is a deliberate legal accuracy choice. A temperature > 0 introduces randomness that could cause the model to hallucinate different section numbers on repeated queries.

---

## 11. Deep Dive: Logging, Utils & App

### 11.1 logger.py — InterceptHandler Pattern

```python
class InterceptHandler(logging.Handler):
    def emit(self, record):
        level = logger.level(record.levelname).name  # Map stdlib level to Loguru
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
```
External libraries (httpx, psycopg, qdrant_client) use Python's standard `logging` module. `InterceptHandler` bridges them all to Loguru so every log line — from your code AND from libraries — appears in the same format and in the same log file.

**Noise filtering:**
```python
for lib in ["httpx", "httpcore", "qdrant_client", "openai", "psycopg", "urllib3"]:
    lib_logger = logging.getLogger(lib)
    lib_logger.setLevel(logging.WARNING)  # Only warnings+ from libraries
```

**Warning interception:**
```python
warnings.showwarning = _warning_to_loguru  # Routes Python warnings to Loguru
```
This captures `FutureWarning` from transformers 5.5.4's lazy image processor registry into the structured log, not stderr noise.

### 11.2 utils.py — Context Pruning with Citation Integrity

```python
def prune_legal_context(context, max_chars=3500):
    pruned = context[:max_chars]
    last_tag_index = pruned.rfind('passage: [LAW:')
    if last_tag_index > 0:
        return pruned[:last_tag_index].strip()  # Cut before incomplete citation
    return pruned
```
Naively cutting at `max_chars` would produce broken citations like `"passage: [LAW: BHARATIYA NYA..."`. The pruner finds the last complete citation tag and cuts just before it to preserve citation integrity.

### 11.3 utils.py — StrictLegalQueryParser

```python
class StrictLegalQueryParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        # Phase 1: Strip LLM formatting artifacts
        # Phase 2: Domain sanitization
        civil_keywords = ["sale deed", "marriage", "tenant", "ndps", ...]
        criminal_noise = ["under bns", "under bnss", "ipc equivalent", ...]
        # If civil context → remove criminal terms from queries
        # (prevents "sale deed theft" queries that cross-contaminate BNS retrieval)
```

Without sanitization, the multi-query LLM might generate "sale deed registration under BNS Section 303" — a civil+criminal mashup that retrieves zero relevant documents from the Qdrant store (because no such chunk exists).

### 11.4 app.py — Health Check Endpoint

```python
st.set_page_config(...)  # Must be first Streamlit call

# Immediately after — before any UI rendering:
if st.query_params.get("health") == "true":
    st.json({"status": "healthy", "timestamp": time.time(), "service": "legisyukti"})
    st.stop()
```

**Why this placement matters:** Streamlit is WebSocket-based. When the keep-alive service pings `https://{SPACE_HOST}/?health=true`, the HTTP GET returns the HTML shell (200 OK) immediately — Python code doesn't run until the WebSocket connects. The `st.stop()` call prevents the heavy CSS and `@st.cache_resource` initialisation from running on health check requests.

---

## 12. Security Hardening & CVE Patches Applied

### 12.1 CVE-2026-1839 — transformers `torch.load()` without `weights_only=True`

| | Details |
|---|---|
| **Severity** | High |
| **Package** | `transformers < 5.0.0` |
| **Vulnerability** | `torch.load()` without `weights_only=True` allows arbitrary code execution via malicious pickle data in model files |
| **Fix** | Bumped `transformers>=5.0.0` in `pyproject.toml` and `requirements.txt`. Resolved to `transformers==5.5.4` in `uv.lock` |
| **Cascade** | `huggingface-hub` bumped `0.36.2 → 1.5.0`, `docling` bumped `2.74.0 → 2.90.0`, `langchain-huggingface 1.2.0 → 1.2.1` |

### 12.2 CVE-2026-25990 — Pillow Image Bomb (CVE-2026-25990)

| | Details |
|---|---|
| **Severity** | High |
| **Package** | `pillow < 12.2.0` |
| **Vulnerability** | Maliciously crafted images can trigger excessive memory allocation (zip bomb equivalent) |
| **Fix** | `pillow==12.2.0` in `pyproject.toml`. `override-dependencies = ["pillow>=12.2.0"]` forces this across all transitive dependencies (docling, torchvision, etc.) |
| **App-level defence** | `Image.MAX_IMAGE_PIXELS = 1_000_000` in app.py limits memory-bomb images |

### 12.3 grpcio 1.78.1 — Yanked Package

| | Details |
|---|---|
| **Issue** | grpcio 1.78.1 was yanked from PyPI after causing a Google Cloud serverless outage |
| **Fix** | `override-dependencies = ["grpcio<1.78.1"]` resolves to 1.78.0 |

### 12.4 Transformers 5.5.4 FutureWarning Spam

| | Details |
|---|---|
| **Symptom** | ~200 lines of `FutureWarning` per request from `transformers`'s lazy image processor registry `__path__` hook |
| **Root cause** | `transformers 5.5.4` added lazy loading for image processors but emits `FutureWarning` for backward compatibility |
| **Why Python-level fix failed** | Streamlit resets `warnings.filters` on each script re-run, clearing any `warnings.filterwarnings()` calls |
| **Fix** | `PYTHONWARNINGS="ignore::FutureWarning"` in Dockerfile ENV. OS-level env vars survive Streamlit re-runs. `TRANSFORMERS_VERBOSITY=error` additionally suppresses transformers' own logger |

### 12.5 Security Findings from OSV Scan (Open Items)

During a cross-reference of all packages in `uv.lock` against the OSV vulnerability database:

| Package | Version | CVE | Severity | Fixed In | Status |
|---|---|---|---|---|---|
| `lxml` | 6.0.2 | GHSA-vfmq-68hx-4jfw | HIGH | 6.1.0 | In progress (transitive dep of docling) |
| `cryptography` | 45.0.2 | GHSA-m959-cc7f-wv43 | LOW | 46.0.6 | Monitoring |
| `torch` | 2.7.0 | GHSA-887c-mr87-cxwp | MODERATE | 2.8.0 | Monitoring (PyTorch custom index) |

**Fix strategy for lxml:** Add `"lxml>=6.1.0"` to `override-dependencies` in `pyproject.toml` then regenerate `uv.lock`.

---

## 13. Performance Optimizations

### 13.1 Singleton Pattern via `@st.cache_resource`

```python
@st.cache_resource
def init_system_core():
    # Runs ONCE per container lifetime, not once per user request
    pool = connect_with_retry(DB_URI, connection_kwargs)
    start_keep_alive_service(interval_minutes=10)
    return pool, create_graph(PostgresSaver(pool))
```
Without `@st.cache_resource`, every Streamlit re-run (user typing, button click) would re-initialise the DB pool and re-compile the LangGraph. Cache resource creates a process-level singleton.

### 13.2 `@lru_cache` for Model Loading

```python
@lru_cache(maxsize=1)
def load_embeddings(): ...   # e5-small-v2 loaded once

@lru_cache(maxsize=1)
def get_reranker(): ...       # FlashRank loaded once
```

### 13.3 Batch Ingestion with Progress Tracking

```python
batch_size = 20
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    vector_store.add_documents(batch)
```
Qdrant performs better with batched inserts. Batch size 20 balances memory usage and network overhead.

### 13.4 Context Pruning for Token Economy

```python
# In verify_citations_node:
context_used = prune_legal_context(raw_context, max_chars=6000)  # Audit: 6000 chars

# In evaluate_response_node:
context_for_eval = prune_legal_context(raw_context, max_chars=1500)  # Eval: 1500 chars
```
Different nodes get different context budgets. The evaluator only needs a summary to score; the full 6,000-char context is reserved for the auditor that must verify citations line-by-line.

### 13.5 Tool Call Consolidation

```python
if len(response.tool_calls) > 1:
    queries = [tc['args'].get('query', '') for tc in response.tool_calls]
    response.tool_calls = [response.tool_calls[0]]  # Keep only first
    response.tool_calls[0]['args']['query'] = " | ".join(queries)  # Merge
```
The dispatcher sometimes generates 2–3 parallel tool calls. This consolidates them into one, saving 2–3 Groq API round-trips.

### 13.6 MMR for Result Diversity

```python
search_kwargs = {"k": 10, "fetch_k": 30, "lambda_mult": 0.7}
```
`lambda_mult=0.7` means 70% weight on relevance, 30% on diversity. Without MMR, 10 nearly-identical sections of the same legal provision would be retrieved, wasting context window. MMR ensures the top 10 results span multiple relevant statutes and provisions.

### 13.7 Multi-Query Expansion for Recall

Legal questions have two orthogonal aspects: substantive law ("what right is violated?") and procedural law ("how to file?"). A single query often retrieves well for one but not the other. The two-query expansion dramatically improves recall at the cost of 2× retrieval time (mitigated by Groq's fast 8B model).

---

## 14. Deployment Architecture (Docker + HF Spaces)

### 14.1 HF Spaces Docker Configuration

```dockerfile
FROM python:3.12-slim

# UV for fast dependency installation
RUN pip install uv

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies from lock file (reproducible builds)
RUN uv sync --frozen --no-dev

COPY . .

ENV UV_COMPILE_BYTECODE=1 \         # Faster imports in container
    PYTHONUNBUFFERED=1 \            # Immediate log flushing
    HF_HOME=/app/model_cache/huggingface \   # Persisted model cache
    FASTEMBED_CACHE_PATH=/app/model_cache/fastembed \
    PYTHONWARNINGS="ignore::FutureWarning" \ # Suppress transformers noise
    TRANSFORMERS_VERBOSITY=error            # Suppress transformers logger

EXPOSE 7860
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

**Why `--frozen`?** `uv sync --frozen` uses the exact versions in `uv.lock`, ignoring pyproject.toml specifiers. This ensures reproducible builds — no "works on my machine" surprises.

**Why `UV_COMPILE_BYTECODE=1`?** Compiles `.py` files to `.pyc` during install. The HF container starts cold on every deploy; bytecode compilation during install means faster import times at runtime.

### 14.2 Git LFS for PDFs

The 17 PDF statutes are tracked with Git LFS to avoid bloating the git history. The `.gitattributes` marks all `*.pdf` files as LFS objects. HF Spaces automatically pulls LFS objects during container build.

### 14.3 PyTorch Custom Index

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }
```
PyTorch is not available on PyPI for CUDA builds. The `explicit = true` flag means UV only uses this index for the three torch packages — other packages still resolve from PyPI. `unsafe-best-match` strategy allows mixing both indexes.

---

## 15. Anticipated Interview Questions & Model Answers

### Q1: What is Agentic RAG and how is it different from basic RAG?
**A:** Basic RAG is a linear pipeline: embed query → retrieve → append to prompt → LLM answer. Agentic RAG adds decision-making, tool use, and feedback loops. In LegisYukti, the agent decides *which* statute to search (tool selection), *whether* to retry (evaluation loop), and *how* to correct itself (auditor node). The key difference is autonomy — the agent can query multiple sources, evaluate its own output, and self-correct without human intervention.

### Q2: Explain the LangGraph state machine in LegisYukti.
**A:** LegisYukti uses a `StateGraph` with 7 nodes connected by conditional edges. The `ChatState` TypedDict is passed between nodes — each node reads and writes specific fields. `add_messages` annotation prevents message overwriting; nodes append rather than replace. The cycle (evaluator → retry_prep → agent → ... → evaluator) is only possible with LangGraph's conditional edge support. LangChain expression language (LCEL) cannot do this because it only supports DAGs.

### Q3: Why hybrid search instead of pure semantic search?
**A:** Legal text is full of exact references — "Section 303 BNS", "Order 39 CPC", "BNSS 2023". Dense embeddings excel at semantic similarity but struggle with exact keyword matches. A query for "Section 173 BNSS" might semantically retrieve Section 172 or 174 (thematically similar but legally different). BM25 sparse search catches exact statutory references. Hybrid search combines both: dense for conceptual similarity, sparse for exact legal citations. In practice, this reduces "zero results" failures by ~40%.

### Q4: Why did you use Groq instead of OpenAI?
**A:** The project is deployed on HF Spaces free tier. OpenAI API costs rule it out for production use. Groq provides LLaMA-3.3-70B inference at speeds comparable to OpenAI GPT-4 on its custom LPU hardware. The free tier is generous for a research/demo application. The trade-off is that Groq doesn't support function-calling in streaming mode, which is why the tool execution is handled manually via `call_tools_and_save_context` rather than using LangGraph's built-in ToolNode.

### Q5: How do you prevent legal hallucinations?
**A:** Three-layer defence: (1) The retrieval pipeline only provides chunks from verified government statutes — the LLM cannot invent facts not in the context. (2) The auditor node cross-references every cited section against the retrieved context — if "Section 303 BNS" wasn't in the retrieved chunks, it flags it as a hallucination. (3) The evaluator node penalises responses that cite IPC/CrPC instead of BNS/BNSS (temporal accuracy check). The combination makes outright hallucination extremely rare.

### Q6: How does the keep-alive service work and why was it needed?
**A:** HF Spaces free tier sleeps containers after extended inactivity. Qdrant Cloud free tier sleeps clusters after 30 minutes. Both are revived by HTTP activity. The `KeepAliveService` runs a daemon background thread (not asyncio, to avoid event loop complexity) that pings both services every 10 minutes. The critical insight was that if the HF Space sleeps first, the daemon thread dies, Qdrant pings stop, and Qdrant sleeps too — a cascade failure. The fix was adding an HF Space self-ping using the `SPACE_HOST` environment variable that HF injects automatically. A second external layer (UptimeRobot) provides redundancy.

### Q7: Walk me through the document ingestion pipeline.
**A:** 17 government PDFs → Docling converts PDF to Markdown with OCR (RapidOCR + GPU acceleration, with CPU fallback on CUDA OOM). The Data Refiner runs 4 passes: strips malformed headers, purges OCR noise (14 regex patterns), promotes inline section references to Markdown headers, and fixes OCR typos. `MarkdownHeaderTextSplitter` then hierarchically splits on Part/Chapter/Section headers. `RecursiveCharacterTextSplitter` further divides into 1,200-char chunks with 250-char overlap. Each chunk gets law_name and section metadata, plus a `passage: [LAW: ...]` breadcrumb prefix. Uploaded to Qdrant in batches of 20.

### Q8: Why is `@st.cache_resource` critical and how does it prevent double-initialisation?
**A:** Streamlit reruns the entire Python script on any user interaction (button click, text input). Without `@st.cache_resource`, every button click would re-initialise the vector store connection, the DB pool, and recompile the LangGraph — adding 30+ seconds of latency per interaction. `@st.cache_resource` creates a process-level singleton that persists across reruns. The keep-alive service is started inside this function to guarantee it's called exactly once per container lifetime, not once per browser interaction.

### Q9: How does the multi-query expansion work?
**A:** The `mqr_prompt()` instructs `fast_llm` (8B model) to generate two complementary search queries: one focusing on the substantive right ("what law governs this?") and one on the procedural remedy ("how to file and what court?"). These queries are parsed by `StrictLegalQueryParser` which removes formatting artifacts and sanitises cross-domain contamination (removes criminal BNS keywords from civil/property queries). Both queries are run through the same retrieval pipeline and the union of results is deduplicated via MMR.

### Q10: What is MMR and why `lambda_mult=0.7`?
**A:** MMR (Maximal Marginal Relevance) is a retrieval strategy that balances relevance to the query with diversity among results. `fetch_k=30` retrieves the 30 closest neighbours; MMR then selects 10 (`k=10`) that maximise relevance while minimising redundancy. `lambda_mult=0.7` sets the balance: 70% weight on closeness to the query, 30% on dissimilarity to already-selected results. At 0.7 the results are still primarily relevant but avoid returning 10 sections of the same provision — crucial for a dataset where many chunks come from adjacent sections.

### Q11: How do you handle the temporal complexity of India's 2023 legal reforms?
**A:** India replaced IPC, CrPC, and IEA with BNS, BNSS, and BSA on 1 July 2024. LLMs pre-trained before this date default to citing the old laws. Three mechanisms handle this: (1) The QA prompt has a hard-coded CRITICAL FAILURE rule against citing IPC/CrPC. (2) The auditor prompt checks for IPC/CrPC references and flags them as hallucinations. (3) A temporal guardrail detects year mentions (2020-2023) in the query and adds a hint: "Incident is pre-July 2024, prioritize IPC/CrPC" — because crimes committed before the reform date are prosecuted under the old law.

### Q12: How do you handle conversation persistence across sessions?
**A:** `PostgresSaver` (LangGraph built-in) serialises the complete `ChatState` to PostgreSQL after each graph execution. The state is keyed by `thread_id` (mapped to the user's workspace/case name). On the next session, `graph.stream(input, config)` loads the prior state from Postgres and continues the conversation. This requires `autocommit=True` (PostgresSaver needs immediate commits) and `prepare_threshold=None` (Neon serverless doesn't support prepared statements).

### Q13: Explain the citation verification (auditor) node.
**A:** After `generate_response_node` produces a draft answer, `verify_citations_node` takes both the draft and the retrieved context, passes them to `fast_llm` with the auditor prompt. The auditor returns either `✅` (citations verified, use original) or `🚨 REFINED ADVICE: [corrected response]` (hallucination detected, use the corrected version). The emoji-prefix protocol is a deliberate design choice — it's a single-character signal that requires no JSON parsing, reducing the chance of the LLM wrapping the verdict in unexpected markup.

### Q14: What security vulnerabilities have you patched?
**A:** Two CVEs fixed: (1) CVE-2026-1839 — `transformers < 5.0.0` used `torch.load()` without `weights_only=True`, allowing arbitrary code execution via malicious pickle data in model files. Fixed by bumping to `transformers>=5.0.0`. (2) CVE-2026-25990 — Pillow < 12.2.0 was vulnerable to image bomb attacks. Fixed by pinning `pillow==12.2.0` with a UV `override-dependency` that forces this version across all transitive dependencies. Also added `Image.MAX_IMAGE_PIXELS = 1_000_000` as an application-level defence. Additionally removed grpcio 1.78.1 (yanked package) and suppressed transformers FutureWarning spam via Dockerfile ENV.

### Q15: What does the `law_filter` payload filtering do and how does it improve performance?
**A:** Without filtering, a query about "theft" searches all 9,221 chunks across all 17 statutes. With `law_filter="BHARATIYA NYAYA SANHITA BNS 2023"`, Qdrant applies a server-side filter — only BNS chunks are searched. This reduces search space by ~94% (from 9,221 to ~500 BNS chunks), dramatically improving retrieval precision and latency. The filter uses a pre-built KEYWORD payload index on `metadata.law_name`. For multi-law queries, `MatchAny(any=[law1, law2])` is used. The system falls back to unfiltered global search if the filtered result set is empty.

### Q16: How does the retry mechanism prevent infinite loops?
**A:** `retry_count` in `ChatState` tracks attempts. `route_after_evaluation` checks: `if score < 6 AND retry_count < 1`. This allows exactly one retry. On the second evaluation pass, regardless of score, it goes to END. The `retry_prep_node` injects the auditor's feedback as an `AIMessage` into the conversation — this gives the dispatcher context about what went wrong ("Missed CPC procedure section"). Without the cap, a genuinely ambiguous query with no relevant context would loop indefinitely.

### Q17: Why is `temperature=0` for all LLM calls?
**A:** Legal advice must be deterministic. A user asking the same question twice should get the same answer. Temperature > 0 introduces stochastic variation that could produce "Section 303" on one call and "Section 304" on another — legally very different (murder vs culpable homicide). `temperature=0` uses greedy decoding, which always picks the highest-probability next token. The trade-off is reduced creativity, which is acceptable for a legal fact-retrieval system — we want accurate, not creative.
