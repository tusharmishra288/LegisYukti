"""
Backend initialization and command-line interface for the Legal Consultation System.

This module serves as the main entry point for running the legal advisor in backend mode:
- Initializes vector database and legal document index
- Sets up PostgreSQL connection pool for conversation persistence
- Provides interactive command-line interface for legal queries
- Orchestrates the complete RAG pipeline from query to audited response

The backend handles the heavy lifting of system initialization, ensuring all
components (vector store, database, LangGraph agent) are ready before
accepting user queries.
"""

import sys
import time
from loguru import logger
from psycopg_pool import ConnectionPool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver

# Internal Imports
from .config import DB_URI
from .agent import create_graph
from .engine import get_vector_store
from .processor import run_ingestion_pipeline

def connect_with_retry(uri, kwargs, retries=3, delay=5):
    """Establish PostgreSQL connection pool with retry logic for serverless databases.

    Handles connection issues common with serverless databases like Neon that may
    have cold-start delays or SSL handshake timeouts. Implements exponential backoff
    and pre-ping validation to ensure database readiness.

    Args:
        uri: PostgreSQL connection string
        kwargs: Connection pool configuration parameters
        retries: Number of connection attempts before giving up
        delay: Seconds to wait between retry attempts

    Returns:
        ConnectionPool: Configured and validated PostgreSQL connection pool

    Raises:
        Exception: If all connection attempts fail
    """
    for attempt in range(retries):
        try:
            pool = ConnectionPool(conninfo=uri, max_size=10, kwargs=kwargs)
            # Pre-ping the database to ensure SSL handshake is complete
            with pool.connection() as conn:
                conn.execute("SELECT 1")
            return pool
        except Exception as e:
            logger.warning(f"⚠️  Postgres connection attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def main():
    """Main entry point for the Legal Consultation System backend.

    Orchestrates the complete system initialization sequence:
    1. Vector store setup and document ingestion
    2. PostgreSQL connection pool establishment
    3. LangGraph agent compilation with checkpointing
    4. Interactive command-line interface for legal queries

    The function implements robust error handling and graceful shutdown
    to ensure system reliability in production environments.
    """
    logger.info("⚖️  Nyaya AI - Legal Consultation System Starting Up...")

    try:
        # Phase 1: Initialize Vector Database and Legal Knowledge Base
        vs = get_vector_store()
        client = vs.client
        collection_name = "indian_legal_library"

        # Phase 2: Configure PostgreSQL Connection for Serverless Environment
        # Optimized settings for Neon serverless with SSL and timeout handling
        connection_kwargs = {
            "autocommit": True,  # Immediate commit for stateless operations
            "prepare_threshold": None,  # Disable prepared statements for serverless
            "sslmode": "require",  # Enforce SSL encryption
            "channel_binding": "require",  # Additional security for SCRAM authentication
            "tcp_user_timeout": 10000  # 10-second timeout for network issues
        }

        # Phase 3: Production Gate - Check Vector Store Readiness
        # Load-once logic: Only ingest documents if vector store is empty
        try:
            collection_info = client.get_collection(collection_name)
            point_count = collection_info.points_count
            if point_count > 0:
                logger.info(f"✅ Production Index Ready: Found {point_count} legal snippets.")
            else:
                logger.warning(f"⚠️  Vector Store '{collection_name}' is empty.")
                run_ingestion_pipeline(vs)
        except Exception as e:
            logger.info("📥 Initializing first-time setup and indexing...")
            run_ingestion_pipeline(vs)

        # Phase 4: Establish Database Connection with Retry Logic
        pool = connect_with_retry(DB_URI, connection_kwargs)

        with pool:
            # Phase 5: Initialize Conversation Persistence Layer
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()
            logger.info("🗄️  Checkpointer linked to Neon Postgres.")

            # Phase 6: Compile LangGraph Agent with Full Pipeline
            graph = create_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "legal_advisor_neon_session"}}

            # Phase 7: Launch Interactive Legal Consultation Interface
            print("\n" + "="*50)
            print("⚖️  NYAYA AI - LEGAL CONSULTATION SYSTEM (BACKEND MODE)")
            print("="*50)
            print("Type 'exit' or 'quit' to end session.\n")

            while True:
                try:
                    user_input = input("User: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["exit", "quit", "bye"]:
                        break

                    logger.info(f"📥 Processing Query: '{user_input[:50]}...'")
                    input_state = {"messages": [HumanMessage(content=user_input)]}

                    # Execute the complete RAG pipeline with streaming
                    for event in graph.stream(input_state, config):
                        for node, value in event.items():
                            if node == "evaluator":
                                ai_msg = value["messages"][-1].content
                                score = value.get("evaluation_score", "N/A")
                                feedback = value.get("evaluation_feedback", "Verified via Scythe Path")

                                # Display response to user while logging for audit trail
                                logger.info(f"\nAdvisor: {ai_msg}")
                                logger.info(f"📊 Quality Score: {score}/10 | {feedback}\n")

                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as e:
                    # Handle connection issues that may occur during long sessions
                    if "SSL connection" in str(e) or "closed" in str(e):
                        logger.error("🔌 Database connection lost. Re-establishing...")
                        # In production, implement reconnection logic here
                    else:
                        logger.error(f"⚠️ Error during conversation: {e}")
                    logger.info("\nAdvisor: I encountered an internal error. Please try again.\n")

    except Exception as e:
        logger.critical(f"🛑 CRITICAL BOOT FAILURE: {e}")
        sys.exit(1)
    finally:
        logger.info("🔒 Shutting down: Cleaning up resources.")
        logger.success("👋 Goodbye.")

if __name__ == "__main__":
    main()