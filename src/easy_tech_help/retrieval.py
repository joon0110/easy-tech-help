"""Small, offline document retrieval for the V1 RAG corpus."""

import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"
STOP_WORDS = {"a", "an", "and", "for", "in", "is", "of", "on", "or", "the", "to"}
GENERIC_TERMS = {
    "email",
    "internet",
    "ios",
    "iphone",
    "message",
    "network",
    "phone",
    "popup",
    "safari",
    "text",
    "wifi",
}


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    category: str
    platform: str
    source_type: str
    tags: tuple[str, ...]
    source_urls: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class SearchHit:
    document: KnowledgeDocument
    score: int


class _FtcArticleParser(HTMLParser):
    """Read the body of the main FTC article, excluding page navigation."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_main = False
        self.in_article = False
        self.div_depth = 0
        self.body_depth: int | None = None
        self.skip_tag: str | None = None
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "main" and attributes.get("id") == "main-content":
            self.in_main = True
        elif tag == "article" and self.in_main:
            self.in_article = "node--view-mode-cfg-default" in classes
        elif tag == "div":
            self.div_depth += 1
            if self.in_article and "field--name-body" in classes:
                self.body_depth = self.div_depth
        if self.body_depth is not None and tag in {
            "script",
            "style",
            "noscript",
            "svg",
        }:
            self.skip_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self.skip_tag:
            self.skip_tag = None
        if tag == "div":
            if self.div_depth == self.body_depth:
                self.body_depth = None
            self.div_depth -= 1
        elif tag == "article":
            self.in_article = False
        elif tag == "main":
            self.in_main = False

    def handle_data(self, data: str) -> None:
        if self.body_depth is not None and self.skip_tag is None:
            self.parts.append(data)


def _read_article(path: Path) -> str:
    if path.suffix != ".html":
        raise ValueError(f"Unsupported source format: {path.name}")
    parser = _FtcArticleParser()
    parser.feed(path.read_text(encoding="utf-8"))
    text = " ".join(" ".join(parser.parts).split())
    if not text:
        raise ValueError(f"Article body not found: {path.name}")
    return text


def _read_document(path: Path, source_type: str) -> str:
    if source_type == "original_html":
        return _read_article(path)
    if source_type == "authored_summary" and path.suffix == ".txt":
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    raise ValueError(f"Invalid or empty document: {path.name}")


def load_documents(directory: Path = DEFAULT_KNOWLEDGE_DIR) -> list[KnowledgeDocument]:
    """Load original FTC articles and clearly labeled Apple-based summaries."""

    catalog = json.loads((directory / "catalog.json").read_text(encoding="utf-8"))
    return [
        KnowledgeDocument(
            id=item["id"],
            title=item["title"],
            category=item["category"],
            platform=item["platform"],
            source_type=item["source_type"],
            tags=tuple(item["tags"]),
            source_urls=tuple(source["url"] for source in item["sources"]),
            text=_read_document(directory / item["file"], item["source_type"]),
        )
        for item in catalog["documents"]
    ]


def _terms(value: str) -> set[str]:
    normalized = value.casefold().replace("wi-fi", "wifi")
    normalized = normalized.replace("e-mail", "email").replace("pop-up", "popup")
    return set(re.findall(r"[a-z0-9]+", normalized)) - STOP_WORDS


def search_documents(
    query: str,
    *,
    category: str | None = None,
    limit: int = 3,
    directory: Path = DEFAULT_KNOWLEDGE_DIR,
) -> list[SearchHit]:
    """Return relevant locally stored source text, strongest match first."""

    if limit < 1:
        raise ValueError("limit must be positive")
    query_terms = _terms(query)
    if not query_terms:
        return []

    hits = []
    for document in load_documents(directory):
        if category is not None and document.category != category:
            continue
        title_terms = _terms(document.title)
        tag_terms = _terms(" ".join(document.tags))
        if not (query_terms - GENERIC_TERMS) & (title_terms | tag_terms):
            continue
        body_terms = _terms(document.text)
        score = sum(
            4 * (term in title_terms) + 3 * (term in tag_terms) + (term in body_terms)
            for term in query_terms
        )
        if score:
            hits.append(SearchHit(document=document, score=score))

    return sorted(hits, key=lambda hit: (-hit.score, hit.document.id))[:limit]
