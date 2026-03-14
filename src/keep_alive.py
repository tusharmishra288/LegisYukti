"""
Keep-Alive Service for Hugging Face Spaces Free Tier

This module implements a background service that prevents both the Hugging Face Space
and Qdrant Cloud from going to sleep during periods of inactivity. This is crucial
for free tier deployments where services automatically sleep after ~15-30 minutes
of inactivity.

The service runs in a background thread and performs periodic health checks:
- Hugging Face Space: Internal health endpoint ping
- Qdrant Cloud: Collection info query to keep the service active

Usage:
    from src.keep_alive import KeepAliveService

    # Start the service
    keep_alive = KeepAliveService()
    keep_alive.start()

    # Stop when shutting down
    keep_alive.stop()
"""

import threading
import time
import requests
from typing import Optional
from src.config import QDRANT_URL, QDRANT_API_KEY
from loguru import logger

class KeepAliveService:
    """
    Background service that keeps cloud services active on free tiers.

    Runs periodic health checks to prevent Hugging Face Spaces and Qdrant Cloud
    from sleeping due to inactivity. Optimized for minimal resource usage while
    maintaining service availability.
    """

    def __init__(self, interval_minutes: int = 10):
        """
        Initialize the keep-alive service.

        Args:
            interval_minutes: How often to ping services (default: 10 minutes)
        """
        self.interval_seconds = interval_minutes * 60
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_ping_time = 0

        # Service endpoints
        self.hf_health_url = "http://localhost:7860/health"  # Streamlit health endpoint
        if QDRANT_URL:
            base_url = QDRANT_URL.rstrip('/')
            self.qdrant_health_url = f"{base_url}/collections"
        else:
            self.qdrant_health_url = None

        logger.info(f"🔄 Keep-Alive Service initialized (interval: {interval_minutes}min)")

    def _ping_huggingface_space(self) -> bool:
        """
        Ping the Hugging Face Space to keep it active.

        Returns:
            bool: True if ping successful, False otherwise
        """
        try:
            # Use a simple GET request to the health endpoint
            response = requests.get(self.hf_health_url, timeout=5)
            if response.status_code == 200:
                logger.debug("✅ Hugging Face Space ping successful")
                return True
            else:
                logger.warning(f"⚠️ Hugging Face Space returned status {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.warning(f"⚠️ Hugging Face Space ping failed: {e}")
            return False

    def _ping_qdrant_cloud(self) -> bool:
        """
        Ping Qdrant Cloud to keep the vector database active.

        Returns:
            bool: True if ping successful, False otherwise
        """
        if not self.qdrant_health_url or not QDRANT_API_KEY:
            logger.debug("ℹ️ Qdrant Cloud not configured, skipping ping")
            return True

        try:
            # Use health endpoint or collections endpoint to keep service active
            headers = {"api-key": QDRANT_API_KEY}
            response = requests.get(self.qdrant_health_url, headers=headers, timeout=10)

            if response.status_code == 200:
                logger.debug("✅ Qdrant Cloud ping successful")
                return True
            else:
                logger.warning(f"⚠️ Qdrant Cloud returned status {response.status_code}")
                return False
        except requests.RequestException as e:
            logger.warning(f"⚠️ Qdrant Cloud ping failed: {e}")
            return False

    def _keep_alive_loop(self):
        """
        Main keep-alive loop that runs in background thread.
        """
        logger.info("🚀 Keep-Alive Service started")

        while self.running:
            try:
                current_time = time.time()

                # Ping both services
                hf_success = self._ping_huggingface_space()
                qdrant_success = self._ping_qdrant_cloud()

                self.last_ping_time = current_time

                # Log status
                if hf_success and qdrant_success:
                    logger.info("🔄 Keep-Alive: Both services pinged successfully")
                elif hf_success:
                    logger.info("🔄 Keep-Alive: HF Space active, Qdrant ping failed")
                elif qdrant_success:
                    logger.info("🔄 Keep-Alive: Qdrant active, HF Space ping failed")
                else:
                    logger.warning("⚠️ Keep-Alive: Both services ping failed")

            except Exception as e:
                logger.error(f"❌ Keep-Alive Service error: {e}")

            # Wait for next interval
            time.sleep(self.interval_seconds)

        logger.info("🛑 Keep-Alive Service stopped")

    def start(self):
        """
        Start the keep-alive service in a background thread.
        """
        if self.running:
            logger.warning("⚠️ Keep-Alive Service is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()

        logger.info("🎯 Keep-Alive Service thread started")

    def stop(self):
        """
        Stop the keep-alive service.
        """
        if not self.running:
            logger.info("ℹ️ Keep-Alive Service is not running")
            return

        logger.info("🛑 Stopping Keep-Alive Service...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)  # Wait up to 5 seconds for clean shutdown

        logger.info("✅ Keep-Alive Service stopped")

    def get_status(self) -> dict:
        """
        Get the current status of the keep-alive service.

        Returns:
            dict: Status information including running state and last ping time
        """
        return {
            "running": self.running,
            "interval_seconds": self.interval_seconds,
            "last_ping_time": self.last_ping_time,
            "time_since_last_ping": time.time() - self.last_ping_time if self.last_ping_time > 0 else None
        }

# Global instance for easy access
_keep_alive_instance: Optional[KeepAliveService] = None

def start_keep_alive_service(interval_minutes: int = 10) -> KeepAliveService:
    """
    Start the global keep-alive service instance.

    Args:
        interval_minutes: Ping interval in minutes

    Returns:
        KeepAliveService: The started service instance
    """
    global _keep_alive_instance

    if _keep_alive_instance is None:
        _keep_alive_instance = KeepAliveService(interval_minutes)

    _keep_alive_instance.start()
    return _keep_alive_instance

def stop_keep_alive_service():
    """
    Stop the global keep-alive service instance.
    """
    global _keep_alive_instance

    if _keep_alive_instance:
        _keep_alive_instance.stop()
        _keep_alive_instance = None

def get_keep_alive_status() -> Optional[dict]:
    """
    Get the status of the global keep-alive service.

    Returns:
        Optional[dict]: Status information or None if not initialized
    """
    global _keep_alive_instance
    return _keep_alive_instance.get_status() if _keep_alive_instance else None