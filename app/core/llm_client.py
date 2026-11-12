"""
OpenAI-compatible client factory.

Credentials are always read from environment variables — never hardcoded.
"""
import logging
import os

from openai import OpenAI

from app.core.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


def get_openai_client(settings: Settings = default_settings) -> OpenAI:
    """
    Return a configured OpenAI-compatible client.

    ``OPENAI_BASE_URL`` is optional: when empty, the SDK default endpoint is used.

    Raises:
        ValueError: If required environment variables are absent.
    """
    settings.validate_api_config()
    base_url = settings.openai_base_url or None
    if base_url is None and not os.environ.get("OPENAI_BASE_URL", "").strip():
        # The SDK reads OPENAI_BASE_URL itself when base_url is None; an *empty* value
        # (e.g. "OPENAI_BASE_URL=" in an env file) would become an empty URL.
        os.environ.pop("OPENAI_BASE_URL", None)
    logger.info("Creating OpenAI-compatible client (custom base_url=%s)", bool(base_url))
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=2,
    )
