"""Reads .env once; every other file uses `from app.config import settings`."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# novadesk/: paths are resolved from here, so it works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing setting '{name}'. Copy .env.example to .env and fill it in.")
    return value.strip()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    llm_model: str
    embed_model: str
    top_k: int
    similarity_threshold: float
    embed_batch_size: int
    embed_max_retries: int
    chroma_path: Path
    chroma_collection: str


settings = Settings(
    google_api_key=_require("GOOGLE_API_KEY"),
    llm_model=_require("LLM_MODEL"),
    embed_model=_require("EMBED_MODEL"),
    top_k=int(_require("TOP_K")),
    similarity_threshold=float(_require("SIMILARITY_THRESHOLD")),
    embed_batch_size=int(_require("EMBED_BATCH_SIZE")),
    embed_max_retries=int(_require("EMBED_MAX_RETRIES")),
    chroma_path=PROJECT_ROOT / _require("CHROMA_PATH"),  # relative to the project root
    chroma_collection=_require("CHROMA_COLLECTION"),
)
