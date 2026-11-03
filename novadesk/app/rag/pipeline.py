"""The /ask pipeline: question -> retrieval -> threshold -> LLM -> answer with sources.

answer_question() reads top to bottom like the spec:
    store.retrieve() -> _apply_threshold() -> _generate() -> _build_sources()
If no chunk passes the threshold we stop BEFORE _generate(): no LLM call.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, model_serializer

from app.config import settings
from app.rag import store
from app.rag.models import get_llm, is_provider_error
from app.rag.prompts import ANSWER_PROMPT, NO_ANSWER, format_context

logger = logging.getLogger(__name__)

_llm = None  # created on first use, then reused


class ProviderUnavailableError(Exception):
    """The AI provider is temporarily unavailable (quota or network). /ask turns this into 503."""


class Source(BaseModel):
    """One entry in the response's `sources` list."""

    type: str  # "guide" / "glossary" / "ticket"
    id: str  # e.g. "04-printers", "glossary", "1043"
    title: str
    section: str | None = None  # guide and glossary only
    resolved_by_name: str | None = None  # tickets only
    score: float

    @model_serializer(mode="wrap")
    def _drop_missing_fields(self, handler):
        """In JSON, leave out fields that don't apply (a ticket has no `section`)."""
        return {key: value for key, value in handler(self).items() if value is not None}


class AnswerResult(BaseModel):
    """What /ask returns (SPEC section 7)."""

    answered: bool
    answer: str | None
    sources: list[Source]
    top_score: float | None  # highest retrieval score, even if it didn't pass the threshold


def answer_question(question: str) -> AnswerResult:
    """Raises ProviderUnavailableError if Gemini can't be reached (quota/network)."""
    with _provider_errors():  # embedding the question calls Gemini
        results = store.retrieve(question)
    top_score = round(results[0].score, 3) if results else None

    relevant = _apply_threshold(results)
    if not relevant:
        logger.info(
            "retrieval: top_score=%s, 0/%d chunks >= %s -> no LLM call",
            top_score, len(results), settings.similarity_threshold,
        )
        return AnswerResult(answered=False, answer=None, sources=[], top_score=top_score)

    logger.info(
        "retrieval: top_score=%s, %d/%d chunks >= %s -> calling LLM",
        top_score, len(relevant), len(results), settings.similarity_threshold,
    )
    with _provider_errors():  # the LLM call
        answer = _generate(question, relevant)
    
    if answer.strip().strip(".") == NO_ANSWER:
        logger.info("top_score=%.3f -> stopped by llm (NO_ANSWER)", top_score)
        return AnswerResult(answered=False, answer=None, sources=[], top_score=top_score)
    return AnswerResult(
        answered=True,
        answer=answer,
        sources=_build_sources(relevant),
        top_score=top_score,
    )


def _apply_threshold(results: list[NodeWithScore]) -> list[NodeWithScore]:
    """Keep only chunks whose similarity is at least SIMILARITY_THRESHOLD."""
    return [r for r in results if r.score >= settings.similarity_threshold]


def _generate(question: str, nodes: list[NodeWithScore]) -> str:
    """Exactly one LLM call: the prompt filled with the relevant chunks and the question."""
    prompt = ANSWER_PROMPT.format(context=format_context(nodes), question=question)
    return _get_llm().complete(prompt).text.strip()


def _build_sources(nodes: list[NodeWithScore]) -> list[Source]:
    """Turn each relevant chunk's metadata into a Source for the response."""
    sources = []
    for result in nodes:
        meta = result.node.metadata
        sources.append(
            Source(
                type=meta["source_type"],
                id=meta["source_id"],
                title=meta["title"],
                section=meta.get("section"),  # .get -> None when the key doesn't exist
                resolved_by_name=meta.get("resolved_by_name"),
                score=round(result.score, 3),
            )
        )
    return sources


@contextmanager
def _provider_errors() -> Iterator[None]:
    """Inside this block, a quota/network error becomes ProviderUnavailableError.

    Any other error (a real bug) passes through unchanged, so it's not hidden behind a 503.
    """
    try:
        yield
    except Exception as exc:
        if is_provider_error(exc):
            logger.warning("provider unavailable: %s", str(exc)[:200])
            raise ProviderUnavailableError() from exc
        raise


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm
