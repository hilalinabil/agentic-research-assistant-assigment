"""
Application configuration module.

Settings are read from environment variables. A local ``.env`` file is loaded
as a convenience for development; variables that are already set in the
process environment (e.g. injected by ``docker run --env-file``) always win,
because ``load_dotenv`` does not override existing variables.
"""
from pathlib import Path
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Base directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(BASE_DIR / ".env")

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _env(name: str, default: str = "") -> str:
    """Read a variable; unset *or empty* values fall back to ``default``."""
    return os.getenv(name, "").strip() or default


class Settings(BaseModel):
    """Application runtime configuration."""

    # --- LLM provider (OpenAI-compatible) -----------------------------------
    openai_api_key: str = Field(default_factory=lambda: _env("OPENAI_API_KEY"))
    # Optional: empty means "use the SDK default / OPENAI_BASE_URL handling".
    openai_base_url: str = Field(default_factory=lambda: _env("OPENAI_BASE_URL"))
    openai_model: str = Field(default_factory=lambda: _env("OPENAI_MODEL"))
    openai_embedding_model: str = Field(default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    llm_timeout_seconds: float = Field(default_factory=lambda: float(_env("LLM_TIMEOUT_SECONDS", "90")))

    # --- Ingestion / retrieval ------------------------------------------------
    corpus_dir: Path = Field(default_factory=lambda: BASE_DIR / "app" / "data" / "corpus")
    target_chunk_size: int = Field(default=600)
    overlap_tokens: int = Field(default=80)
    embedding_batch_size: int = Field(default=64)
    default_top_k: int = Field(default=5)

    # --- Agent limits -----------------------------------------------------------
    max_agent_iterations: int = Field(default_factory=lambda: int(_env("MAX_AGENT_ITERATIONS", "6")))
    max_tool_calls: int = Field(default=12)
    max_search_top_k: int = Field(default=8)
    max_evidence_chars: int = Field(default=60_000)

    # --- Frontend -----------------------------------------------------------------
    frontend_dist_dir: Path = Field(default_factory=lambda: BASE_DIR / "frontend" / "dist")

    def missing_api_config(self) -> list[str]:
        """Return the names of required variables that are missing."""
        missing = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.openai_model:
            missing.append("OPENAI_MODEL")
        if not self.openai_embedding_model:
            missing.append("OPENAI_EMBEDDING_MODEL")
        return missing

    def validate_api_config(self) -> None:
        """Raise ``ValueError`` if a required LLM setting is missing."""
        missing = self.missing_api_config()
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Singleton instance
settings = Settings()
