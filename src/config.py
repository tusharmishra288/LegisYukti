"""
Configuration module for the Nyaya AI -Legal Consultation System.

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
# Auto-detect CUDA for GPU acceleration in embeddings and processing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Selection ---
# High-quality embedding model for legal text semantic search
EMBED_MODEL_ID = "intfloat/e5-small-v2"

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

# --- LLM Model Configurations ---
# Primary legal reasoning model: High-capability for complex legal analysis
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,  # Deterministic responses for legal accuracy
    api_key=GROQ_API_KEY,
    max_tokens=800,  # Sufficient for detailed legal explanations
    max_retries=2
)

# Fast utility model: Optimized for quick query expansion and internal processing
fast_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,  # Consistent query transformations
    api_key=GROQ_API_KEY,
    max_tokens=1024,  # Longer context for query expansion
    max_retries=2
)