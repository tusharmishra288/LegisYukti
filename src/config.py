"""
Configuration module for the Agentic RAG Framework System.

Centralizes all system configuration including:
- Directory structure and paths
- Hardware detection and model selection
- API authentication and service endpoints
- LLM model configurations optimized for legal reasoning
"""

import os
import torch
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# --- Directory Structure ---
# Project root and core directories for legal document processing
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR = PROJECT_ROOT / "docs"  # Source legal PDFs
SCRATCH_DIR = PROJECT_ROOT / "scratch"  # Intermediate processing files
CACHE_DIR = PROJECT_ROOT / "model_cache"  # Model weights and embeddings
LOG_DIR = PROJECT_ROOT / "logs"  # Application logs

# Ensure all required directories exist
for d in [DOCS_DIR, SCRATCH_DIR, CACHE_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# Load environment variables from project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# --- Model Caching Configuration ---
# Redirect model caches to project directory for portability
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["FASTEMBED_CACHE_PATH"] = str(CACHE_DIR / "fastembed")

# --- Hardware Detection ---
# Order of preference: an explicit DEVICE_TYPE override, then CUDA (Linux/Windows
# with an NVIDIA card), then MPS (Apple Silicon), then CPU.
#
# DEVICE_TYPE was already being set in docker-compose.yml but nothing ever read
# it; it is now the supported override. An override naming an unavailable backend
# warns and falls back rather than crashing, so a .env copied between machines
# does not break the app.
def _detect_device() -> str:
    requested = (os.getenv("DEVICE_TYPE") or "").strip().lower()

    def available(name: str) -> bool:
        if name == "cuda":
            return torch.cuda.is_available()
        if name == "mps":
            # Apple Silicon GPU. getattr guards torch builds without the backend.
            mps = getattr(torch.backends, "mps", None)
            return bool(mps and mps.is_available() and mps.is_built())
        return name == "cpu"

    if requested:
        if requested in ("cuda", "mps", "cpu") and available(requested):
            return requested
        logger.warning(
            f"⚠️ DEVICE_TYPE={requested!r} requested but unavailable on this machine - auto-detecting."
        )
    for candidate in ("cuda", "mps"):
        if available(candidate):
            return candidate
    return "cpu"


DEVICE = _detect_device()

# --- Model Selection ---
# High-quality embedding model for legal text semantic search
EMBED_MODEL_ID = "intfloat/e5-small-v2"

# Groq model IDs. Overridable from .env because Groq retires models on a rolling
# basis - a deprecation should be a config change, not a code change. Check what
# your account can actually serve with:
#   curl -s https://api.groq.com/openai/v1/models \
#        -H "Authorization: Bearer $GROQ_API_KEY" | jq -r '.data[].id' | sort
# Hardcoded values ship with the image and reach Hugging Face; the env override
# is there so a Groq retirement can be worked around from Space settings without
# a redeploy. Blank or absent both fall through to the default.
GROQ_MODEL_PRIMARY = (os.getenv("GROQ_MODEL_PRIMARY") or "").strip() or "openai/gpt-oss-120b"
GROQ_MODEL_FAST = (os.getenv("GROQ_MODEL_FAST") or "").strip() or "openai/gpt-oss-20b"

# Fallback message when no relevant legal context is found
NO_CONTEXT_MSG = "I could not find any legally verified references for this specific query."

# --- API Authentication ---
# Required API keys for external services
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # For gated HuggingFace models
QDRANT_URL = os.getenv("QDRANT_URL")  # Vector database endpoint
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Vector database authentication
DB_URI = os.getenv("POSTGRES_URI")  # Conversation persistence database

# Authentication validation with helpful error messages
if not GROQ_API_KEY:
    logger.critical("❌ GROQ_API_KEY is missing from .env!")
if not HF_TOKEN:
    logger.warning("⚠️ HF_TOKEN not found. Gated models may be inaccessible.")
else:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    logger.success("🔑 HuggingFace Token authenticated.")

logger.info(f"🖥️  Hardware Acceleration: {DEVICE.upper()} detected.")
if DEVICE == "mps":
    # FlashRank runs on ONNX Runtime (CPU only), so reranking does not benefit.
    logger.info("🍎 Apple Silicon GPU in use for embeddings. Note: FlashRank reranking stays on CPU.")

# --- LLM Model Configurations ---
# Primary legal reasoning model: High-capability for complex legal analysis
llm = ChatGroq(
    model_name=GROQ_MODEL_PRIMARY,
    temperature=0,  # Deterministic responses for legal accuracy
    api_key=GROQ_API_KEY,
    max_tokens=800,  # Sufficient for detailed legal explanations
    max_retries=2
)

# Fast utility model: Optimized for quick query expansion and internal processing
fast_llm = ChatGroq(
    model_name=GROQ_MODEL_FAST,
    temperature=0,  # Consistent query transformations
    api_key=GROQ_API_KEY,
    max_tokens=1024,  # Longer context for query expansion
    max_retries=2
)