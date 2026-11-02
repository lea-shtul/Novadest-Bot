"""All the text the LLM sees: the answer prompt and how chunks are presented to it.

The prompt is in Hebrew because the bot answers employees in Hebrew.
In English, the rules say: answer only from the chunks below; never use general
knowledge or invent system names, extensions or paths; if the chunks only partly
answer, say what is known and refer the rest to IT at extension 7700; on conflicting
sources, an official guide beats a ticket, and a newer ticket beats an older one;
end with the sources used; be short and clear, with numbered steps where relevant.
"""

from llama_index.core.schema import NodeWithScore

ANSWER_PROMPT = """אתה בוט התמיכה הטכנית של נובה טק. אתה עונה לעובדי החברה בעברית.

כללים:
1. ענה רק על סמך הקטעים שלמטה. אל תשתמש בידע כללי, ואל תמציא שמות של מערכות, שלוחות או נתיבים.
2. אם הקטעים לא עונים על השאלה במלואה, אמור מה כן ידוע, ושאר הפרטים כדאי לברר מול צוות IT בשלוחה 7700.
3. אם יש סתירה בין מקורות: מדריך רשמי גובר על טיקט, ובין שני טיקטים, החדש גובר.
4. בסוף התשובה ציין את המקורות שהשתמשת בהם, למשל: "לפי המדריך 'מדפסות', ולפי טיקט 1043 שנפתר ע״י דני".
5. כתוב בקצרה ובבהירות. כשיש שלבים, כתוב אותם כרשימה ממוספרת.

הקטעים:
{context}

השאלה: {question}"""


def format_context(nodes: list[NodeWithScore]) -> str:
    """Build the {context} block: each chunk = a source line + its text, separated by a blank line."""
    return "\n\n".join(
        f"{_source_line(result.node.metadata)}\n{result.node.text}" for result in nodes
    )


def _source_line(metadata: dict) -> str:
    """The bracketed line that tells the LLM where a chunk comes from (and how recent it is)."""
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
