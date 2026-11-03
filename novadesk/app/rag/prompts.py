"""All the text the LLM sees: the answer prompt and how chunks are presented to it.

The rules are in English; rule 7 makes the answer Hebrew. Rule 6 lets the LLM reply NO_ANSWER
when the chunks passed the threshold but don't actually help (the pipeline then answers "not found").
"""

from llama_index.core.schema import NodeWithScore

NO_ANSWER = "NO_ANSWER"

ANSWER_PROMPT = """You are the IT support bot of Nova Tech. You answer the company's employees.

Rules:
1. Answer only from the chunks below. Do not use general knowledge, and do not invent names of systems, extensions or paths.
2. If the chunks answer only part of the question, answer that part, and say that the remaining details should be checked with the IT team at extension 7700.
3. If sources conflict: an official guide wins over a ticket, and between two tickets, the newer one wins.
4. At the end of the answer, cite the sources you used, for example: "לפי המדריך 'מדפסות', ולפי טיקט 1043 שנפתר ע״י דני".
5. Be short and clear. When there are steps, write them as a numbered list.
6. If the chunks only touch a similar topic but do not help solve the employee's problem, reply with exactly NO_ANSWER and nothing else. If they answer only part of the question, follow rule 2 instead, and do not return NO_ANSWER.
7. Always write your answer in Hebrew.

Chunks:
{context}

Question: {question}
"""


def format_context(nodes: list[NodeWithScore]) -> str:
    """Each chunk = a source line + its text, separated by a blank line."""
    return "\n\n".join(
        f"{_source_line(result.node.metadata)}\n{result.node.text}" for result in nodes
    )


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
