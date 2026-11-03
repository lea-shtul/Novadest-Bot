"""Source documents -> chunks (TextNodes): one per `##` section of a guide or glossary, one per ticket."""

from pathlib import Path

import frontmatter
from llama_index.core.schema import TextNode


def guide_to_nodes(path: Path) -> list[TextNode]:
    meta, body = _read_markdown(path)
    guide_id = path.stem  # e.g. "04-printers"

    nodes = []
    for number, (heading, text) in enumerate(_split_sections(body), start=1):
        metadata = {
            "source_type": "guide",
            "source_id": guide_id,
            "title": meta["title"],
            "section": heading,
            "category": meta["category"],
            "date": meta["updated"],
        }
        header = f"מדריך: {meta['title']} | סעיף: {heading}"
        nodes.append(_make_node(f"guide-{guide_id}-{number}", f"{header}\n{text}", metadata))
    return nodes


def glossary_to_nodes(path: Path) -> list[TextNode]:
    meta, body = _read_markdown(path)

    nodes = []
    for number, (term, text) in enumerate(_split_sections(body), start=1):
        metadata = {
            "source_type": "glossary",
            "source_id": "glossary",
            "title": meta["title"],
            "section": term,
            "category": meta["category"],
            "date": meta["updated"],
        }
        header = f"מילון מונחים | מונח: {term}"
        nodes.append(_make_node(f"glossary-{number}", f"{header}\n{text}", metadata))
    return nodes


def ticket_to_node(ticket: dict, resolved_by_name: str) -> TextNode:
    text = (
        f"טיקט {ticket['id']}: {ticket['title']}\n"
        f"בעיה: {ticket['problem']}\n"
        f"פתרון: {ticket['solution']}"
    )
    metadata = {
        "source_type": "ticket",
        "source_id": str(ticket["id"]),
        "title": ticket["title"],
        "category": ticket["category"],
        "date": ticket["resolved_at"],
        "resolved_by_name": resolved_by_name,
    }
    return _make_node(f"ticket-{ticket['id']}", text, metadata)


def _read_markdown(path: Path) -> tuple[dict, str]:
    """(title/category/updated from the frontmatter, body)."""
    post = frontmatter.load(path)
    for key in ("title", "category", "updated"):
        if key not in post.metadata:
            raise ValueError(f"{path.name}: frontmatter is missing '{key}'")
    # str(): YAML turns 2026-10-14 into a date object, and Chroma metadata needs plain strings.
    meta = {key: str(post[key]) for key in ("title", "category", "updated")}
    return meta, post.content


def _split_sections(body: str) -> list[tuple[str, str]]:
    """[(heading, text), ...] for each `## ` section; anything before the first one is dropped."""
    sections = []
    heading, lines = None, []
    for line in body.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def _make_node(node_id: str, text: str, metadata: dict) -> TextNode:
    # Metadata is for us (sources), not for the embedding or the LLM: the header already gives context.
    return TextNode(
        id_=node_id,
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
        excluded_llm_metadata_keys=list(metadata),
    )
