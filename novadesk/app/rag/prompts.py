"""All the text the LLM sees: the answer prompt, the rewrite prompt, and how chunks are presented.

The rules are in English; rule 7 makes the answer Hebrew. Rule 6 lets the LLM reply NO_ANSWER
when the chunks passed the threshold but don't actually help (the pipeline then answers "not found").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.schema import NodeWithScore

if TYPE_CHECKING:  # only for the type hint; a real import would be circular (pipeline imports us)
    from app.rag.pipeline import Turn

NO_ANSWER = "NO_ANSWER"

SYSTEM_PROMPT = """You are the IT support bot of Nova Tech. You answer the company's employees.

Rules:
1. Answer only from the chunks below. Do not use general knowledge, and do not invent names of systems, extensions or paths.
2. If the chunks answer only part of the question, answer that part, and say that the remaining details should be checked with the IT team at extension 7700.
3. If sources conflict: an official guide wins over a ticket, and between two tickets, the newer one wins.
4. At the end of the answer, cite the sources you used, for example: "לפי המדריך 'מדפסות', ולפי טיקט 1043 שנפתר ע״י דני".
5. Be short and clear. When there are steps, write them as a numbered list.
6. If the chunks only touch a similar topic but do not help solve the employee's problem, reply with exactly NO_ANSWER and nothing else. If they answer only part of the question, follow rule 2 instead, and do not return NO_ANSWER.
7. Always write your answer in Hebrew.
"""

USER_PROMPT = """<chunks>
{context}
</chunks>

<question>
{question}
</question>
"""

# Used before retrieval when there is a conversation: a follow-up like "and if that doesn't help?"
# has no topic of its own, so searching Chroma with it would find nothing useful.
REWRITE_PROMPT = """Below is a conversation between an employee and Nova Tech's IT support bot, and then the employee's new question.

Rewrite the new question so it can be understood on its own, without the conversation: replace words like "it", "this" or "that one" with what they refer to, and add the topic from the conversation if it is missing.

Rules:
1. If the new question is already understandable on its own, or is about a different topic than the conversation, return it exactly as it is.
2. Do not answer the question. Return only the question, with no explanations and no quotes.
3. Write the question in Hebrew.

<conversation>
{history}
</conversation>

<new_question>
{question}
</new_question>
"""


def format_context(nodes: list[NodeWithScore]) -> str:
    """Each chunk = a source line + its text, separated by a blank line."""
    return "\n\n".join(
        f"{_source_line(result.node.metadata)}\n{result.node.text}" for result in nodes
    )


def format_history(turns: list[Turn]) -> str:
    """The conversation as plain text for REWRITE_PROMPT: one "employee/bot" pair per round."""
    return "\n\n".join(f"Employee: {turn.question}\nBot: {turn.answer}" for turn in turns)


def _source_line(metadata: dict) -> str:
    # Includes the date, so the LLM can apply rule 3 (newer ticket wins).
    source_type = metadata["source_type"]
    if source_type == "guide":
        return f"[מקור: מדריך | {metadata['source_id']} | עודכן {metadata['date']}]"
    if source_type == "ticket":
        return (
            f"[מקור: טיקט {metadata['source_id']} | "
            f"נפתר ע״י {metadata['resolved_by_name']} | {metadata['date']}]"
        )
    if source_type == "glossary":
        return f"[מקור: מילון מונחים | {metadata['section']}]"
    raise ValueError(f"Unknown source_type: {source_type}")