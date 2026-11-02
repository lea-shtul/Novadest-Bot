"""The gatekeeper: ALL access to Chroma goes through this file.

Nobody else imports chromadb. Other files call get_index / reset_knowledge /
add_nodes / retrieve and never see collections, distances or clients.
"""

import math
import time

import chromadb
from chromadb.errors import NotFoundError
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings
from app.rag.models import get_embed_model, is_provider_error

# Created on first use, then reused (the "single connection").
_client = None
_index = None


def get_index() -> VectorStoreIndex:
    """The one VectorStoreIndex over our Chroma collection (built once, then cached)."""
    global _index
    if _index is None:
        vector_store = ChromaVectorStore(chroma_collection=_get_collection())
        _index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=get_embed_model()
        )
    return _index


def reset_knowledge() -> None:
    """Delete the collection (if it exists) and create it again, empty, with cosine."""
    global _index
    try:
        _get_client().delete_collection(settings.chroma_collection)
    except NotFoundError:
        pass  # first run: nothing to delete
    _index = None  # the old index pointed at the deleted collection
    _get_collection()  # recreate it now, empty


def add_nodes(nodes: list[TextNode]) -> None:
    """Embed and insert nodes in batches; on a provider error, wait and retry with backoff."""
    index = get_index()
    size = settings.embed_batch_size
    batches = [nodes[i : i + size] for i in range(0, len(nodes), size)]

    for number, batch in enumerate(batches, start=1):
        for attempt in range(settings.embed_max_retries + 1):  # first try + retries
            try:
                index.insert_nodes(batch)  # embeds the batch (one API call), then writes to Chroma
                break
            except Exception as exc:
                if not is_provider_error(exc) or attempt == settings.embed_max_retries:
                    raise  # a real bug, or we've run out of retries: stop and show the error
                wait = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32, 64... seconds
                print(f"  batch {number}: provider error ({str(exc)[:100]}...). Waiting {wait}s, then retrying")
                time.sleep(wait)
        print(f"  batch {number}/{len(batches)} done ({len(batch)} chunks)")


def retrieve(question: str) -> list[NodeWithScore]:
    """Return the TOP_K closest chunks, each with a cosine similarity score (higher = closer).

    We use a Retriever, not a query engine: a retriever only searches, it never calls the LLM.
    """
    retriever = get_index().as_retriever(similarity_top_k=settings.top_k)
    results = retriever.retrieve(question)
    for result in results:
        result.score = _to_similarity(result.score)
    return results


# ---------------------------------------------------------------- helpers


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return _client


def _get_collection():
    """Get (or create) our collection with cosine distance, and make sure it really is cosine."""
    collection = _get_client().get_or_create_collection(
        settings.chroma_collection,
        configuration={"hnsw": {"space": "cosine"}},
    )
    # get_or_create IGNORES `configuration` if the collection already exists,
    # so an old collection could silently still be L2. Check instead of trusting.
    space = (collection.configuration.get("hnsw") or {}).get("space")
    if space != "cosine":
        raise RuntimeError(
            f"Collection '{settings.chroma_collection}' uses '{space}' distance, not cosine. "
            "Run `python -m app.ingest` to rebuild it."
        )
    return collection


def _to_similarity(score: float) -> float:
    """Undo LlamaIndex's conversion and return cosine similarity.

    ChromaVectorStore reports score = exp(-distance). So:
        distance   = -ln(score)
        similarity = 1 - distance = 1 + ln(score)
    """
    return 1 + math.log(score)
