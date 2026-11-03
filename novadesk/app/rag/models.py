"""The only file that knows we use Gemini: the LLM, the embedding model, and provider errors."""

from functools import cache

import httpx
from google.genai.errors import APIError
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from app.config import settings

# gemini-embedding-2 ignores task_type, so Google recommends putting the task in the text.
_QUERY_PREFIX = "task: search result | query: "
_DOCUMENT_PREFIX = "title: none | text: "
_QUERY_TASK_TYPE = "RETRIEVAL_QUERY"  # what LlamaIndex passes when it embeds a question

# Quota or server trouble at the provider: temporary, not a bug.
_PROVIDER_ERROR_CODES = {429, 500, 502, 503, 504}


@cache  # created on first use, then reused
def get_llm() -> GoogleGenAI:
    # No retries: on a quota/network error the user gets "try again in a minute" right away.
    return GoogleGenAI(model=settings.llm_model, api_key=settings.google_api_key, max_retries=0)


def get_embed_model() -> GoogleGenAIEmbedding:
    # retries=1 means one attempt: ingest does its own retries, questions fail fast.
    return _PrefixedGeminiEmbedding(
        model_name=settings.embed_model, api_key=settings.google_api_key, retries=1
    )


def is_provider_error(exc: BaseException) -> bool:
    """A Gemini quota/server error or a network failure (httpx), as opposed to a bug."""
    if isinstance(exc, APIError):
        return exc.code in _PROVIDER_ERROR_CODES
    return isinstance(exc, httpx.TransportError)


class _PrefixedGeminiEmbedding(GoogleGenAIEmbedding):
    # Every embedding call in the parent goes through these two methods, with the task_type.

    def _embed_texts(self, texts, task_type=None):
        return super()._embed_texts(_add_prefix(texts, task_type), task_type)

    async def _aembed_texts(self, texts, task_type=None):
        return await super()._aembed_texts(_add_prefix(texts, task_type), task_type)



#פונקציה שמקבלת חלק- טקסט ומדביקה לו את המדבקה, אם זה שאלה- שאלה, אם זה טיקט או מידע דאטה שלי - דאטה
#הכל עובר דרך הפונקציה הזו aembed_texts ולכן דרסתי אותה וכל טקסט שיעבור אמבדינג יקבל מדבקה
#בלי מדבקות, Gemini מחפש טקסט שנשמע כמו השאלה. עם מדבקות, הוא מחפש טקסט שעונה על השאלה.
def _add_prefix(texts: list[str], task_type: str | None) -> list[str]:
    prefix = _QUERY_PREFIX if task_type == _QUERY_TASK_TYPE else _DOCUMENT_PREFIX
    return [prefix + text for text in texts]
