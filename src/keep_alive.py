"""
Keep-Alive Service for HF Spaces + Qdrant Cloud Free Tier

Runs a background daemon thread that periodically pings:
  1. The HF Space itself (via SPACE_HOST env var injected by HF) — prevents container sleep
  2. Qdrant Cloud (via /collections endpoint) — prevents cluster inactivity timeout

Without pinging the HF Space, the Docker container shuts down, the daemon thread
dies, Qdrant pings stop, and both services go inactive regardless of this module.
"""

import os
import threading
import time
import requests
from typing import Optional
from src.config import QDRANT_URL, QDRANT_API_KEY
from loguru import logger

_RETRY_DELAY_SECONDS = 60  # Retry quickly after a failed ping before waiting the full interval


class KeepAliveService:
    """Background service that keeps HF Space and Qdrant Cloud active on free tiers."""

    def __init__(self, interval_minutes: int = 10):
        self.interval_seconds = interval_minutes * 60
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_ping_time = 0

        # HF Spaces injects SPACE_HOST automatically (e.g. "username-spacename.hf.space")
        space_host = os.getenv("SPACE_HOST")
        self.hf_health_url = f"https://{space_host}/?health=true" if space_host else None

        if QDRANT_URL:
            self.qdrant_health_url = f"{QDRANT_URL.rstrip('/')}/collections"
        else:
            self.qdrant_health_url = None

        logger.info(
            f"🔄 Keep-Alive Service initialized (interval: {interval_minutes}min) | "
            f"HF Space: {'✅' if self.hf_health_url else '⚠️ SPACE_HOST not set'} | "
            f"Qdrant: {'✅' if self.qdrant_health_url else '⚠️ not configured'}"
        )

    def _ping_hf_space(self) -> bool:
        if not self.hf_health_url:
            logger.debug("ℹ️ SPACE_HOST not set — skipping HF Space ping")
            return True
        try:
            response = requests.get(self.hf_health_url, timeout=20)
            if response.status_code == 200:
                logger.debug("✅ HF Space ping successful")
                return True
            logger.warning(f"⚠️ HF Space returned status {response.status_code}")
            return False
        except requests.RequestException as e:
            logger.warning(f"⚠️ HF Space ping failed: {e}")
            return False

    def _ping_qdrant_cloud(self) -> bool:
        if not self.qdrant_health_url or not QDRANT_API_KEY:
            logger.debug("ℹ️ Qdrant not configured — skipping ping")
            return True
        try:
            headers = {"api-key": QDRANT_API_KEY, "User-Agent": "LegisYukti-KeepAlive/1.0"}
            response = requests.get(self.qdrant_health_url, headers=headers, timeout=15)
            if response.status_code == 200:
                logger.debug("✅ Qdrant Cloud ping successful")
                return True
            logger.warning(f"⚠️ Qdrant Cloud returned status {response.status_code}")
            return False
        except requests.RequestException as e:
            logger.warning(f"⚠️ Qdrant Cloud ping failed: {e}")
            return False

    def _keep_alive_loop(self):
        logger.info("🚀 Keep-Alive Service started")

        while self.running:
            try:
                hf_ok = self._ping_hf_space()
                qdrant_ok = self._ping_qdrant_cloud()
                self.last_ping_time = time.time()

                if hf_ok and qdrant_ok:
                    logger.info("🔄 Keep-Alive: all services pinged successfully")
                else:
                    logger.warning(
                        f"⚠️ Keep-Alive partial failure — HF Space: {hf_ok}, Qdrant: {qdrant_ok}. "
                        f"Retrying in {_RETRY_DELAY_SECONDS}s"
                    )
                    time.sleep(_RETRY_DELAY_SECONDS)
                    continue  # Retry immediately instead of waiting the full interval

            except Exception as e:
                logger.error(f"❌ Keep-Alive Service error: {e}")

            time.sleep(self.interval_seconds)

        logger.info("🛑 Keep-Alive Service stopped")

    def start(self):
        # Check both the flag AND whether the thread is actually alive to handle resurrection
        if self.running and self.thread and self.thread.is_alive():
            logger.warning("⚠️ Keep-Alive Service is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()
        logger.info("🎯 Keep-Alive Service thread started")

    def stop(self):
        if not self.running:
            logger.info("ℹ️ Keep-Alive Service is not running")
            return

        logger.info("🛑 Stopping Keep-Alive Service...")
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        logger.info("✅ Keep-Alive Service stopped")

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "interval_seconds": self.interval_seconds,
            "last_ping_time": self.last_ping_time,
            "time_since_last_ping": time.time() - self.last_ping_time if self.last_ping_time > 0 else None,
            "hf_space_url": self.hf_health_url,
            "qdrant_url": self.qdrant_health_url,
        }


# Global singleton
_keep_alive_instance: Optional[KeepAliveService] = None


def start_keep_alive_service(interval_minutes: int = 10) -> KeepAliveService:
    global _keep_alive_instance

    if _keep_alive_instance is None:
        _keep_alive_instance = KeepAliveService(interval_minutes)

    _keep_alive_instance.start()
    return _keep_alive_instance


def stop_keep_alive_service():
    global _keep_alive_instance

    if _keep_alive_instance:
        _keep_alive_instance.stop()
        _keep_alive_instance = None


def get_keep_alive_status() -> Optional[dict]:
    global _keep_alive_instance
    return _keep_alive_instance.get_status() if _keep_alive_instance else None
