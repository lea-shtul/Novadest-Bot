"""question -> retrieval -> threshold -> LLM -> answer with sources.

If no chunk passes the threshold, we stop before the LLM: no LLM call.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from app.config import settings
from app.rag import store
from app.rag.models import get_llm, is_provider_error
from app.rag.prompts import ANSWER_PROMPT, NO_ANSWER, format_context

logger = logging.getLogger(__name__)


class ProviderUnavailableError(Exception):
    """The AI provider is temporarily unavailable (quota or network): "try again in a minute"."""


@dataclass
class Source:
    type: str  # "guide" / "glossary" / "ticket"
    id: str  # e.g. "04-printers", "glossary", "1043"
    title: str
    section: str | None  # guide and glossary only
    resolved_by_name: str | None  # tickets only
    score: float


@dataclass
class AnswerResult:
    answered: bool
    answer: str | None
    sources: list[Source]
    top_score: float | None  # highest retrieval score, even if it didn't pass the threshold


def answer_question(question: str) -> AnswerResult:
    """Raises ProviderUnavailableError if Gemini can't be reached (quota/network)."""
    with _provider_errors():  # embedding the question calls Gemini
        results = store.retrieve(question)
    top_score = round(results[0].score, 3) if results else None
    relevant = [r for r in results if r.score >= settings.similarity_threshold]

    logger.info(
        "retrieval: top_score=%s, %d/%d chunks >= %s -> %s",
        top_score, len(relevant), len(results), settings.similarity_threshold,
        "calling LLM" if relevant else "no LLM call",
    )
    if not relevant:
        return AnswerResult(answered=False, answer=None, sources=[], top_score=top_score)

    prompt = ANSWER_PROMPT.format(context=format_context(relevant), question=question)
    with _provider_errors():  # the one LLM call
        answer = get_llm().complete(prompt).text.strip()

    if answer.strip(".") == NO_ANSWER:
        logger.info("top_score=%.3f -> stopped by llm (NO_ANSWER)", top_score)
        return AnswerResult(answered=False, answer=None, sources=[], top_score=top_score)

    sources = [
        Source(
            type=r.node.metadata["source_type"],
            id=r.node.metadata["source_id"],
            title=r.node.metadata["title"],
            section=r.node.metadata.get("section"),
            resolved_by_name=r.node.metadata.get("resolved_by_name"),
            score=round(r.score, 3),
        )
        for r in relevant
    ]
    return AnswerResult(answered=True, answer=answer, sources=sources, top_score=top_score)


@contextmanager
def _provider_errors() -> Iterator[None]:
    """Turn quota/network errors into ProviderUnavailableError; a real bug passes through unchanged."""
    try:
        yield
    except Exception as exc:
        if is_provider_error(exc):
            logger.warning("provider unavailable: %s", str(exc)[:200])
            raise ProviderUnavailableError() from exc
        raise
