"""Threshold experiment (SPEC section 8): which SIMILARITY_THRESHOLD separates the two groups?

Retrieval only (no LLM) for every question in data/test_questions.json, then a table sorted
by top score and a suggested threshold range.
Run from the project root:  python -m tools.threshold_experiment
"""

import json
import sys

from app.config import PROJECT_ROOT, settings
from app.rag import store


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # so Hebrew prints correctly in PowerShell
    data = json.loads((PROJECT_ROOT / "data" / "test_questions.json").read_text(encoding="utf-8"))

    rows = []  # (top score, group, top chunk id, question)
    for group in ("answerable", "unanswerable"):
        for item in data[group]:
            top = store.retrieve(item["question"])[0]
            rows.append((top.score, group, top.node.id_, item["question"]))
    rows.sort(key=lambda row: row[0], reverse=True)

    # The question goes last: Hebrew is right-to-left, so a column after it wouldn't line up.
    print(f"\n{'score':<7}{'group':<14}{'top chunk':<34}question")
    print("-" * 100)
    for score, group, chunk_id, question in rows:
        print(f"{score:<7.3f}{group:<14}{chunk_id:<34}{question}")

    answerable = [score for score, group, _, _ in rows if group == "answerable"]
    unanswerable = [score for score, group, _, _ in rows if group == "unanswerable"]
    low, high = min(answerable), max(unanswerable)
    print()
    print(f"Lowest top score, answerable:     {low:.3f}")
    print(f"Highest top score, unanswerable:  {high:.3f}")

    # A chunk counts when score >= threshold, so a separating T has: high < T <= low.
    if low > high:
        print(
            f"\nThe groups separate. Any threshold T with {high:.3f} < T <= {low:.3f} works;"
            f"\nthe middle of the gap, {(low + high) / 2:.3f}, leaves the most room on both sides."
        )
    else:
        print("\nThe groups OVERLAP: no single threshold separates them. Questions in the overlap zone:")
        for score, group, _, question in rows:
            if low <= score <= high:
                print(f"  {score:.3f}  {group:<14}{question}")
        print(
            "Any threshold here is a trade-off: higher = fewer wrong answers to unanswerable questions,"
            "\nbut more answerable questions sent to IT."
        )

    current = settings.similarity_threshold
    missed = sum(score < current for score in answerable)
    let_through = sum(score >= current for score in unanswerable)
    print(
        f"\nCurrent SIMILARITY_THRESHOLD={current}: {missed}/{len(answerable)} answerable questions rejected, "
        f"{let_through}/{len(unanswerable)} unanswerable questions let through."
    )


if __name__ == "__main__":
    main()
