"""The translator: builds the LLM and the embedding model from .env.

This is the ONLY file that knows we use Gemini. Everything else just asks for
"an LLM", "an embedding model", or "was this a provider error?".
"""

import httpx
from google.genai.errors import APIError
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from app.config import settings

# gemini-embedding-2 ignores task_type, so Google recommends putting the task in the text.
_QUERY_PREFIX = "task: search result | query: "
_DOCUMENT_PREFIX = "title: none | text: "

# The task_type the LlamaIndex integration passes when it embeds a question.
_QUERY_TASK_TYPE = "RETRIEVAL_QUERY"

# HTTP codes that mean "the provider is temporarily unavailable" (quota or server trouble).
_PROVIDER_ERROR_CODES = {429, 500, 502, 503, 504}


def get_llm() -> GoogleGenAI:
    """The chat model that writes answers."""
    return GoogleGenAI(
        model=settings.llm_model,
        api_key=settings.google_api_key,
        # No built-in retries: on a quota/network error /ask fails fast with 503
        # ("try again in a minute") instead of keeping the user waiting.
        max_retries=0,
    )


def get_embed_model() -> GoogleGenAIEmbedding:
    """The embedding model, with the task prefixes built in."""
    return _PrefixedGeminiEmbedding(
        model_name=settings.embed_model,
        api_key=settings.google_api_key,
        # retries=1 means "one attempt, no retries": ingest retries in store.add_nodes,
        # and /ask fails fast with 503.
        retries=1,
    )


def is_provider_error(exc: BaseException) -> bool:
    """True if exc is a Gemini quota/server error or a network failure (worth waiting and retrying)."""
    if isinstance(exc, APIError):
        return exc.code in _PROVIDER_ERROR_CODES
    # The Gemini client talks HTTP through httpx; TransportError covers
    # connection failures and timeouts.
    return isinstance(exc, httpx.TransportError)


class _PrefixedGeminiEmbedding(GoogleGenAIEmbedding):
    """GoogleGenAIEmbedding that writes the task into the text itself.

    Every embedding call in the parent class (query/document, single/batch,
    sync/async) goes through _embed_texts or _aembed_texts, with a task_type
    telling us whether it's a question or a document. We add the matching
    prefix and then let the parent do the actual API call.
    """

    def _embed_texts(self, texts, task_type=None):
        return super()._embed_texts(_add_prefix(texts, task_type), task_type)

    async def _aembed_texts(self, texts, task_type=None):
        return await super()._aembed_texts(_add_prefix(texts, task_type), task_type)


def _add_prefix(texts: list[str], task_type: str | None) -> list[str]:
    prefix = _QUERY_PREFIX if task_type == _QUERY_TASK_TYPE else _DOCUMENT_PREFIX
    return [prefix + text for text in texts]
