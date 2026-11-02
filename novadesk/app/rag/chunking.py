"""The scissors: turns source documents into chunks (LlamaIndex TextNodes).

- Guides and the glossary: one chunk per `##` section, with a Hebrew context
  header at the top of the text. Sections that are too long are split further.
- Tickets: exactly one chunk per ticket (problem + solution together).

This file knows nothing about Gemini or Chroma. It only reads files and builds nodes.
"""

from pathlib import Path

import frontmatter
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.utils import get_tokenizer

from app.config import settings

# The same tokenizer SentenceSplitter uses by default (tiktoken), so our
# "does it fit?" check and the splitter agree on what a token is.
# For Hebrew this is an approximation of Gemini's real token count.
_count_tokens = lambda text: len(get_tokenizer()(text))


# ---------------------------------------------------------------- public API


def guide_to_nodes(path: Path) -> list[TextNode]:
    """One guide file -> one node per `##` section (or several, if a section is too long)."""
    meta, body = _read_markdown(path)
    guide_id = path.stem  # filename without extension, e.g. "04-printers"

    nodes = []
    for number, (heading, text) in enumerate(_split_sections(body), start=1):
        header = f"מדריך: {meta['title']} | סעיף: {heading}"
        metadata = {
            "source_type": "guide",
            "source_id": guide_id,
            "title": meta["title"],
            "section": heading,
            "category": meta["category"],
            "date": meta["updated"],
        }
        nodes += _section_to_nodes(header, text, f"guide-{guide_id}-{number}", metadata)
    return nodes


def glossary_to_nodes(path: Path) -> list[TextNode]:
    """The glossary file -> one node per term (each term is a `##` section)."""
    meta, body = _read_markdown(path)

    nodes = []
    for number, (term, text) in enumerate(_split_sections(body), start=1):
        header = f"מילון מונחים | מונח: {term}"
        metadata = {
            "source_type": "glossary",
            "source_id": "glossary",
            "title": meta["title"],
            "section": term,
            "category": meta["category"],
            "date": meta["updated"],
        }
        nodes += _section_to_nodes(header, text, f"glossary-{number}", metadata)
    return nodes


def ticket_to_node(ticket: dict, resolved_by_name: str) -> TextNode:
    """One resolved ticket -> exactly one node. The caller supplies the technician's name."""
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


# ---------------------------------------------------------------- helpers


def _read_markdown(path: Path) -> tuple[dict, str]:
    """Parse the YAML frontmatter and return (title/category/updated, body without frontmatter)."""
    post = frontmatter.load(path)
    for key in ("title", "category", "updated"):
        if key not in post.metadata:
            raise ValueError(f"{path.name}: frontmatter is missing '{key}'")

    meta = {
        "title": str(post["title"]),
        "category": str(post["category"]),
        # YAML turns 2026-10-14 into a date object; Chroma metadata needs a plain string.
        "updated": str(post["updated"]),
    }
    return meta, post.content


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split Markdown on `## ` headings -> [(heading, section text), ...].

    Anything before the first `##` (the `#` document title) is not a section and is dropped.
    """
    sections = []
    heading, lines = None, []

    for line in body.splitlines():
        if line.startswith("## "):
            if heading is not None:  # close the previous section
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line[3:].strip(), []
        elif heading is not None:  # we're inside a section
            lines.append(line)

    if heading is not None:  # close the last section
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def _section_to_nodes(header: str, text: str, base_id: str, metadata: dict) -> list[TextNode]:
    """The safety net: one node if the section fits, otherwise split into parts.

    Every part gets the context header, so each part still says where it came from.
    """
    full_text = f"{header}\n{text}"
    if _count_tokens(full_text) <= settings.chunk_max_tokens:
        return [_make_node(base_id, full_text, metadata)]

    # Leave room for the header (+1 for the newline) inside the token budget.
    budget = settings.chunk_max_tokens - _count_tokens(header) - 1
    if budget <= settings.chunk_overlap_tokens:
        raise ValueError(
            f"{base_id}: CHUNK_MAX_TOKENS={settings.chunk_max_tokens} is too small. "
            f"The context header alone takes {_count_tokens(header)} tokens, leaving "
            f"{budget} for text, which must be more than CHUNK_OVERLAP_TOKENS."
        )
    splitter = SentenceSplitter(
        chunk_size=budget,
        chunk_overlap=settings.chunk_overlap_tokens,
    )
    parts = splitter.split_text(text)
    return [
        _make_node(f"{base_id}-part{n}", f"{header}\n{part}", metadata)
        for n, part in enumerate(parts, start=1)
    ]


def _make_node(node_id: str, text: str, metadata: dict) -> TextNode:
    """Build one TextNode.

    By default LlamaIndex glues the metadata onto the text when it embeds the node
    and when it sends it to the LLM. We don't want that: metadata is for us (sources,
    filtering), and the context header already gives the text its context. So we
    exclude every metadata key from both.
    """
    return TextNode(
        id_=node_id,
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
        excluded_llm_metadata_keys=list(metadata),
    )
