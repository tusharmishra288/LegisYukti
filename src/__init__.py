"""
Initialization of the LegisYukti - Agentic RAG Framework codebase, setting up imports and logging.
"""
from loguru import logger 
from .config import *
from .processor import *
from .engine import *
from .agent import *
from .logger import *
from .processor import *
from .prompts import *
from .utils import *

# Logging the package initialization
logger.info("📦 LegisYukti System codebase initialized.")