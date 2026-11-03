"""Deletes the knowledge base and rebuilds it from data/. Run with:  python -m app.ingest

A script, not an endpoint, on purpose: rebuilding everything must not be reachable over the network.
"""

import json
import sys
from collections import Counter

from llama_index.core.schema import TextNode

from app.config import PROJECT_ROOT, settings
from app.rag import store
from app.rag.chunking import glossary_to_nodes, guide_to_nodes, ticket_to_node

DATA = PROJECT_ROOT / "data"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # so Hebrew prints correctly in PowerShell

    print("1/4 Resetting the knowledge base...")
    store.reset_knowledge()

    print("2/4 Chunking documents...")
    nodes = _collect_nodes()
    print(f"  {len(nodes)} chunks")

    print("3/4 Embedding and storing...")
    store.add_nodes(nodes)

    print("4/4 Summary")
    counts = Counter(n.metadata["source_type"] for n in nodes)
    for source_type in ("guide", "glossary", "ticket"):
        print(f"  {source_type}: {counts[source_type]}")
    print(f"  total: {len(nodes)}")
    print(f"  stored in: {settings.chroma_path} (exists: {settings.chroma_path.exists()})")


def _collect_nodes() -> list[TextNode]:
    """All guides, the glossary, and the tickets marked added_to_knowledge."""
    nodes = []
    for path in sorted((DATA / "guides").glob("*.md")):
        nodes += guide_to_nodes(path)
    nodes += glossary_to_nodes(DATA / "glossary.md")

    users = json.loads((DATA / "seed_users.json").read_text(encoding="utf-8"))
    names = {user["id"]: user["name"] for user in users}  # technician id -> name
    tickets = json.loads((DATA / "seed_tickets.json").read_text(encoding="utf-8"))
    for ticket in tickets:
        if ticket.get("added_to_knowledge") is True:
            nodes.append(ticket_to_node(ticket, names[ticket["resolved_by"]]))
    return nodes


if __name__ == "__main__":
    main()
