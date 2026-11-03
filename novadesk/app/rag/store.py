"""All access to Chroma goes through this file; nobody else imports chromadb."""

import math
import time
from functools import cache

import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings
from app.rag.models import get_embed_model, is_provider_error


@cache  # created on first use, then reused
def get_index() -> VectorStoreIndex:
    vector_store = ChromaVectorStore(chroma_collection=_get_collection())
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=get_embed_model())


def reset_knowledge() -> None:
    """Delete the collection and recreate it, empty, with cosine distance."""
    try:
        _get_client().delete_collection(settings.chroma_collection)
    except NotFoundError:
        pass  # first run: nothing to delete
    get_index.cache_clear()  # the cached index pointed at the deleted collection
    _get_collection()


def add_nodes(nodes: list[TextNode]) -> None:
    """Embed and insert in batches; on a provider error, wait and retry (2, 4, 8... seconds)."""
    index = get_index()
    size = settings.embed_batch_size
    batches = [nodes[i : i + size] for i in range(0, len(nodes), size)]

    for number, batch in enumerate(batches, start=1):
        for attempt in range(settings.embed_max_retries + 1):  # first try + retries
            try:
                index.insert_nodes(batch)  # one embedding API call, then the write to Chroma
                break
            except Exception as exc:
                if not is_provider_error(exc) or attempt == settings.embed_max_retries:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  batch {number}: provider error ({str(exc)[:100]}...). Waiting {wait}s, then retrying")
                time.sleep(wait)
        print(f"  batch {number}/{len(batches)} done ({len(batch)} chunks)")


def retrieve(question: str) -> list[NodeWithScore]:
    """The TOP_K closest chunks with cosine similarity scores. A retriever never calls the LLM."""
    results = get_index().as_retriever(similarity_top_k=settings.top_k).retrieve(question)
    for result in results:
        result.score = _to_similarity(result.score)
    return results


@cache
def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(settings.chroma_path))


def _get_collection():
    collection = _get_client().get_or_create_collection(
        settings.chroma_collection, configuration={"hnsw": {"space": "cosine"}}
    )
    # get_or_create ignores `configuration` for an existing collection, which could still be L2.
    space = (collection.configuration.get("hnsw") or {}).get("space")
    if space != "cosine":
        raise RuntimeError(
            f"Collection '{settings.chroma_collection}' uses '{space}' distance, not cosine. "
            "Run `python -m app.ingest` to rebuild it."
        )
    return collection


def _to_similarity(score: float) -> float:
    """LlamaIndex reports exp(-distance); cosine similarity = 1 - distance = 1 + ln(score)."""
    return 1 + math.log(score)
