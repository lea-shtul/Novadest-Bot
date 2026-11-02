"""Retrieval only: show the TOP_K chunks closest to a question, with their scores.

No LLM call. Useful for debugging what the bot "sees" for a question.
Run with:  python -m app.search "יוצאים דפים לבנים מהמדפסת בקומה 3"
"""

import sys

from llama_index.core.schema import NodeWithScore

from app.rag import store

_PREVIEW_CHARS = 150


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # so Hebrew prints correctly in PowerShell
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python -m app.search "your question"')
        sys.exit(1)

    print(f"Question: {question}\n")
    for rank, result in enumerate(store.retrieve(question), start=1):
        _print_result(rank, result)


def _print_result(rank: int, result: NodeWithScore) -> None:
    meta = result.node.metadata
    where = meta["title"] + (f" | {meta['section']}" if "section" in meta else "")
    preview = result.node.text.replace("\n", " ")[:_PREVIEW_CHARS]
    print(f"#{rank}  score={result.score:.3f}  {result.node.id_}  ({meta['source_type']})")
    print(f"    {where}")
    print(f"    {preview}...\n")


if __name__ == "__main__":
    main()
