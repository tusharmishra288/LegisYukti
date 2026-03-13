"""
Centralized logging configuration for the Nyaya AI - Legal Consultation System.

This module provides unified logging across all components with:
- Structured console output for development
- Comprehensive file logging for debugging
- Universal interception of standard library logs
- Noise filtering for clean operation
- Warning capture for database and API issues
"""

import sys
import logging
import warnings
from loguru import logger
from .config import LOG_DIR

class InterceptHandler(logging.Handler):
    """Bridge handler that forwards standard logging to Loguru.

    Captures all logging.getLogger() calls from libraries and routes them
    through Loguru's structured logging system for consistent formatting
    and filtering across the entire application.
    """
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging():
    """Initialize comprehensive logging system for the legal advisor.

    Sets up a dual-output logging system:
    1. Clean console output for development and monitoring
    2. Detailed file logging for debugging and audit trails

    Also intercepts all standard library logging and external library noise
    to maintain clean operation and comprehensive observability.

    Returns:
        Logger: Configured Loguru logger instance
    """
# Clear any existing handlers to prevent double-logging
    logging.root.handlers = []
    logger.remove()
    
    # Console handler: Clean, readable output for development
    logger.add(
        sys.stderr, 
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>", 
        level="INFO"
    )
    
    # File handler: Comprehensive logging for debugging and compliance
    logger.add(
        LOG_DIR / "legaladvisorsystem.log", 
        rotation="50 MB",  # Rotate when file reaches 50MB
        retention="10 days",  # Keep logs for 10 days
        compression="zip",  # Compress old logs
        level="DEBUG",  # Capture all log levels
        enqueue=True,  # Thread-safe async logging
        backtrace=True,  # Include full stack traces
        diagnose=True  # Include variable values in errors
    )

    # Universal interception: Capture all standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Noise filtering: Silence verbose external libraries
    noisy_libraries = ["httpx", "httpcore", "qdrant_client", "openai", "psycopg", "urllib3", "nicegui"]
    

    for lib in noisy_libraries:
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.WARNING)
        lib_logger.handlers = [InterceptHandler()] # Route through our handler
        lib_logger.propagate = False

    # Warning interception: Capture database and API warnings
    def _warning_to_loguru(message, category, filename, lineno, file=None, line=None):
        logger.warning(
            f"⚠️ {category.__name__} detected at {filename}:{lineno} -> {message}"
        )

    warnings.showwarning = _warning_to_loguru

    logger.success("🛡️  Universal Logger Active: Capturing all System, API, and Library logs.")
    return logger

log = setup_logging()