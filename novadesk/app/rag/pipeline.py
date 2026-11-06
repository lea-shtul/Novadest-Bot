"""question (+ conversation) -> rewrite -> retrieval -> threshold -> LLM -> answer with sources.

If there is a conversation, the question is first rewritten to stand on its own. From then on
only that standalone question is used, for the search and for the answer: the conversation itself
is never shown to the answer LLM. If no chunk passes the threshold, we stop before the answer call.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from llama_index.core.llms import ChatMessage

from app.config import settings
from app.rag import store
from app.rag.models import get_llm, is_provider_error
from app.rag.prompts import (
    NO_ANSWER,
    REWRITE_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
    format_context,
    format_history,
)

logger = logging.getLogger(__name__)


class ProviderUnavailableError(Exception):
    """The AI provider is temporarily unavailable (quota or network): "try again in a minute"."""


@dataclass
class Turn:
    """One round of the conversation: the employee's question and the bot's answer."""

    question: str
    answer: str


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
    search_query: str  # what we actually searched with (the rewritten question, or the original)


def answer_question(question: str, history: list[Turn] | None = None) -> AnswerResult:
    """Answer one question, using the last rounds of the conversation (if any).

    The pipeline keeps no state: whoever calls it (the UI, later the server) holds the history.
    Raises ProviderUnavailableError if Gemini can't be reached (quota/network).
    """
    recent = _recent_turns(history)

    # 1. Rewrite: only when there is a conversation. A first question is searched as is.
    search_query = _rewrite_question(question, recent) if recent else question

    # 2. Retrieval, with the standalone question.
    with _provider_errors():  # embedding the question calls Gemini
        results = store.retrieve(search_query)
    top_score = round(results[0].score, 3) if results else None
    relevant = [r for r in results if r.score >= settings.similarity_threshold]

    logger.info(
        "retrieval: top_score=%s, %d/%d chunks >= %s -> %s",
        top_score, len(relevant), len(results), settings.similarity_threshold,
        "calling LLM" if relevant else "no LLM call",
    )
    if not relevant:
        return AnswerResult(answered=False, answer=None, sources=[], top_score=top_score, search_query=search_query)

    # 3. Answer, with the standalone question only. When the whole conversation was sent here,
    # it pulled the LLM toward the earlier topic: after a talk about VPN from home, "my printer
    # at home doesn't work" was answered with office-printer steps instead of NO_ANSWER.
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=USER_PROMPT.format(context=format_context(relevant), question=search_query)),
    ]
    with _provider_errors():  # the answer LLM call
        answer = get_llm().chat(messages).message.content.strip()

    if answer.strip(".") == NO_ANSWER:
        logger.info("top_score=%.3f -> stopped by llm (NO_ANSWER)", top_score)
        return AnswerResult(answered=False, answer=None, sources=[], top_score=top_score, search_query=search_query)

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
    return AnswerResult(answered=True, answer=answer, sources=sources, top_score=top_score, search_query=search_query)


def _recent_turns(history: list[Turn] | None) -> list[Turn]:
    """The last HISTORY_TURNS rounds. Careful: history[-0:] would return everything, not nothing."""
    if not history or settings.history_turns <= 0:
        return []
    return history[-settings.history_turns:]


def _rewrite_question(question: str, turns: list[Turn]) -> str:
    """Turn a follow-up into a standalone question. An unrelated question comes back unchanged."""
    prompt = REWRITE_PROMPT.format(history=format_history(turns), question=question)
    with _provider_errors():  # one short LLM call
        rewritten = get_llm().chat([ChatMessage(role="user", content=prompt)]).message.content

    rewritten = rewritten.strip().strip('"')
    if not rewritten:  # an empty reply should never cost the employee an answer
        return question
    logger.info("rewrite: %r -> %r", question, rewritten)
    return rewritten


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