"""
Phase 0 validation tests: Environment, Configuration, and Corpus integrity.
"""
import os
from pathlib import Path
import pytest
from dotenv import load_dotenv

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


@pytest.mark.integration
def test_required_env_variables_present():
    """Verify that the required LLM environment variables are set and non-empty.

    Integration-only: CI and fresh clones have no credentials.
    OPENAI_BASE_URL is optional (the SDK default is used when it is empty).
    """
    required_keys = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ]
    for key in required_keys:
        value = os.getenv(key)
        assert value is not None and len(value.strip()) > 0, f"Missing or empty environment variable: {key}"


def test_gitignore_protects_env():
    """Ensure that .env is explicitly excluded in .gitignore."""
    gitignore_path = BASE_DIR / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"
    
    ignored_patterns = [
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert ".env" in ignored_patterns or "/.env" in ignored_patterns, ".env must be listed in .gitignore"


def test_env_example_matches_required_keys():
    """Verify .env.example contains all required variable names without values."""
    env_example_path = BASE_DIR / ".env.example"
    assert env_example_path.exists(), ".env.example must exist"
    
    example_keys = set()
    for line in env_example_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            example_keys.add(key.strip())
            assert val.strip() == "", f".env.example should not contain values for {key}"
            
    required_keys = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    }
    assert required_keys.issubset(example_keys), f"Missing keys in .env.example: {required_keys - example_keys}"


def test_corpus_files_exist():
    """Verify that the local corpus directory exists and contains the expected documents."""
    corpus_dir = BASE_DIR / "app" / "data" / "corpus"
    assert corpus_dir.exists() and corpus_dir.is_dir(), f"Corpus directory not found at {corpus_dir}"
    
    files = list(corpus_dir.glob("*.md"))
    assert len(files) >= 20, f"Expected at least 20 markdown files in corpus, found {len(files)}"
    
    file_names = {f.name for f in files}
    expected_samples = {
        "dailymed-warfarin.md",
        "dailymed-apixaban.md",
        "dailymed-metformin.md",
        "dailymed-sertraline.md",
        "GLOSSARY.md",
    }
    assert expected_samples.issubset(file_names), f"Missing critical corpus files: {expected_samples - file_names}"


def test_core_dependencies_importable():
    """Verify that core libraries are installed and importable in Python 3.13."""
    import openai
    import numpy as np
    import fastapi
    import pydantic
    import uvicorn
    import httpx

    assert int(np.__version__.split(".")[0]) >= 2, "NumPy 2.x+ required for Python 3.13 support"
    assert int(pydantic.__version__.split(".")[0]) >= 2, "Pydantic v2 required"


def test_settings_defaults_for_empty_optional_values(monkeypatch):
    """Empty optional variables (e.g. copied from .env.example) must fall back to defaults, not crash."""
    from app.core.config import Settings
    for name in ("OPENAI_EMBEDDING_MODEL", "LLM_TIMEOUT_SECONDS", "MAX_AGENT_ITERATIONS", "OPENAI_BASE_URL"):
        monkeypatch.setenv(name, "")
    s = Settings()
    assert s.openai_embedding_model == "text-embedding-3-small"
    assert s.llm_timeout_seconds == 90 and s.max_agent_iterations == 6 and s.openai_base_url == ""


def test_settings_reports_missing_chat_model(monkeypatch):
    from app.core.config import Settings
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_MODEL", "")
    s = Settings()
    assert s.missing_api_config() == ["OPENAI_MODEL"]
    with pytest.raises(ValueError, match="OPENAI_MODEL"):
        s.validate_api_config()


def test_client_with_empty_base_url_uses_sdk_default(monkeypatch):
    from app.core.config import Settings
    from app.core.llm_client import get_openai_client
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    client = get_openai_client(Settings(openai_api_key="k", openai_model="m", openai_base_url=""))
    assert str(client.base_url).startswith("https://")


def test_client_uses_configured_base_url():
    from app.core.config import Settings
    from app.core.llm_client import get_openai_client
    client = get_openai_client(Settings(openai_api_key="k", openai_model="m", openai_base_url="http://proxy:4000/v1"))
    assert str(client.base_url).startswith("http://proxy:4000/v1")
