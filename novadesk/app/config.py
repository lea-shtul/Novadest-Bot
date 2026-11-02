"""Reads .env once and exposes every tunable value through a single Settings object.

Every other file does `from app.config import settings` instead of reading
environment variables itself, so all configuration lives in one place.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# The project root is the folder above app/ (i.e. novadesk/).
# We resolve paths from here, so the app works no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load novadesk/.env into the process environment (os.environ).
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    """Return the value of an environment variable, or fail with a clear message if it's missing."""
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(
            f"Missing setting '{name}'. Copy .env.example to .env and fill it in."
        )
    return value.strip()


@dataclass(frozen=True)  # frozen = settings can't be changed by accident at runtime
class Settings:
    # Gemini
    google_api_key: str
    llm_model: str
    embed_model: str

    # Retrieval
    top_k: int
    similarity_threshold: float

    # Chunking
    chunk_max_tokens: int
    chunk_overlap_tokens: int

    # Ingest
    embed_batch_size: int
    embed_max_retries: int

    # Chroma
    chroma_path: Path
    chroma_collection: str

    # CORS
    cors_origins: list[str]


def _load_settings() -> Settings:
    """Read every key from the environment and convert it to the right type."""
    return Settings(
        google_api_key=_require("GOOGLE_API_KEY"),
        llm_model=_require("LLM_MODEL"),
        embed_model=_require("EMBED_MODEL"),
        top_k=int(_require("TOP_K")),
        similarity_threshold=float(_require("SIMILARITY_THRESHOLD")),
        chunk_max_tokens=int(_require("CHUNK_MAX_TOKENS")),
        chunk_overlap_tokens=int(_require("CHUNK_OVERLAP_TOKENS")),
        embed_batch_size=int(_require("EMBED_BATCH_SIZE")),
        embed_max_retries=int(_require("EMBED_MAX_RETRIES")),
        # A relative path in .env is taken relative to the project root.
        chroma_path=PROJECT_ROOT / _require("CHROMA_PATH"),
        chroma_collection=_require("CHROMA_COLLECTION"),
        # "a,b,c" -> ["a", "b", "c"], ignoring empty entries and extra spaces
        cors_origins=[o.strip() for o in _require("CORS_ORIGINS").split(",") if o.strip()],
    )


# Created once, when this module is first imported.
settings = _load_settings()
