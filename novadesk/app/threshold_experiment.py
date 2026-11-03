"""Threshold experiment (SPEC section 8): which SIMILARITY_THRESHOLD separates the two groups?

Reads data/test_questions.json, runs retrieval only (no LLM call) for every question,
and prints a table sorted by top score, followed by a suggested threshold range.
Run with:  python -m app.threshold_experiment
"""

import json
import sys
import time
from dataclasses import dataclass

from app.config import PROJECT_ROOT, settings
from app.rag import store
from app.rag.models import is_provider_error

QUESTIONS_FILE = PROJECT_ROOT / "data" / "test_questions.json"
GROUPS = ("answerable", "unanswerable")  # the keys in the JSON file

# The free tier allows ~100 embeddings per minute; on a quota error, wait and try again.
_MAX_ATTEMPTS = 4
_WAIT_SECONDS = 20


@dataclass
class Row:
    question: str
    group: str  # "answerable" / "unanswerable"
    top_score: float
    top_chunk_id: str


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # so Hebrew prints correctly in PowerShell
    data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    rows = []
    for group in GROUPS:
        for item in data[group]:
            rows.append(_measure(item["question"], group))
    rows.sort(key=lambda row: row.top_score, reverse=True)

    _print_table(rows)
    _print_summary(rows)


def _measure(question: str, group: str) -> Row:
    """Retrieval only: the closest chunk and its score. No LLM call."""
    top = _retrieve_with_retry(question)[0]
    return Row(question, group, top.score, top.node.id_)


def _retrieve_with_retry(question: str):
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return store.retrieve(question)
        except Exception as exc:
            if not is_provider_error(exc) or attempt == _MAX_ATTEMPTS:
                raise
            print(f"  provider error, waiting {_WAIT_SECONDS}s and retrying ({attempt}/{_MAX_ATTEMPTS - 1})...")
            time.sleep(_WAIT_SECONDS)


def _print_table(rows: list[Row]) -> None:
    # The question goes last: Hebrew is right-to-left, so a fixed-width column after it wouldn't line up.
    print(f"\n{'score':<7}{'group':<14}{'top chunk':<34}question")
    print("-" * 100)
    for row in rows:
        print(f"{row.top_score:<7.3f}{row.group:<14}{row.top_chunk_id:<34}{row.question}")


def _print_summary(rows: list[Row]) -> None:
    answerable = [row for row in rows if row.group == "answerable"]
    unanswerable = [row for row in rows if row.group == "unanswerable"]
    lowest_answerable = min(row.top_score for row in answerable)
    highest_unanswerable = max(row.top_score for row in unanswerable)

    print()
    print(f"Lowest top score, answerable:     {lowest_answerable:.3f}")
    print(f"Highest top score, unanswerable:  {highest_unanswerable:.3f}")

    # A chunk counts as relevant when score >= threshold, so a separating threshold T must be
    # above every unanswerable score and at most the lowest answerable score.
    if lowest_answerable > highest_unanswerable:
        middle = (lowest_answerable + highest_unanswerable) / 2
        print(
            f"\nThe groups separate. Any threshold T with {highest_unanswerable:.3f} < T <= {lowest_answerable:.3f} works;"
            f"\nthe middle of the gap, {middle:.3f}, leaves the most room on both sides."
        )
    else:
        print("\nThe groups OVERLAP: no single threshold separates them. Questions in the overlap zone:")
        for row in rows:
            if lowest_answerable <= row.top_score <= highest_unanswerable:
                print(f"  {row.top_score:.3f}  {row.group:<14}{row.question}")
        print(
            "Any threshold here is a trade-off: higher = fewer wrong answers to unanswerable questions,"
            "\nbut more answerable questions sent to IT."
        )

    # How the current setting does, for comparison.
    current = settings.similarity_threshold
    missed = sum(row.top_score < current for row in answerable)
    let_through = sum(row.top_score >= current for row in unanswerable)
    print(
        f"\nCurrent SIMILARITY_THRESHOLD={current}: {missed}/{len(answerable)} answerable questions rejected, "
        f"{let_through}/{len(unanswerable)} unanswerable questions let through."
    )


if __name__ == "__main__":
    main()
