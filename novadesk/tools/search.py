"""Retrieval only (no LLM): the TOP_K chunks closest to a question, with their scores.

Run from the project root:  python -m tools.search "יוצאים דפים לבנים מהמדפסת בקומה 3"
"""

import sys

from app.rag import store


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # so Hebrew prints correctly in PowerShell
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python -m tools.search "your question"')
        sys.exit(1)

    print(f"Question: {question}\n")
    for rank, result in enumerate(store.retrieve(question), start=1):
        meta = result.node.metadata
        where = meta["title"] + (f" | {meta['section']}" if "section" in meta else "")
        preview = result.node.text.replace("\n", " ")[:150]
        print(f"#{rank}  score={result.score:.3f}  {result.node.id_}  ({meta['source_type']})")
        print(f"    {where}")
        print(f"    {preview}...\n")


if __name__ == "__main__":
    main()
