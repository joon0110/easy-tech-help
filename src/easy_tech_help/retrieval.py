"""Load and search local reference documents."""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from itertools import pairwise
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
        if self.body_depth is not None and tag in {"p", "li", "h2", "h3", "br"}:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if self.body_depth is not None and tag in {"p", "li", "h2", "h3"}:
            self.parts.append("\n\n")
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
    text = "\n\n".join(
        " ".join(part.split())
        for part in " ".join(parser.parts).split("\n\n")
        if part.strip()
    )
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


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    document: KnowledgeDocument
    text: str


@dataclass(frozen=True)
class ChunkHit:
    chunk: KnowledgeChunk
    score: float


def build_chunks(
    documents: list[KnowledgeDocument], max_words: int = 75
) -> list[KnowledgeChunk]:
    """Literal paragraph excerpts split at sentence boundaries when possible.

    A sentence longer than the budget remains intact: preserving a condition or
    consequence is more useful than satisfying a hard word limit.
    """
    if max_words < 1:
        raise ValueError("max_words must be positive")
    chunks = []
    for doc in documents:
        ordinal = 0
        # Group adjacent short paragraphs so headings and questions retain context.
        # Final excerpts remain literal slices, including paragraph breaks.
        paragraphs = doc.text.split("\n\n")
        grouped = []
        pending = []
        for paragraph in paragraphs:
            pending.append(paragraph)
            if len(" ".join(pending).split()) >= 45 and paragraph.rstrip().endswith(
                (".", "!", "?")
            ):
                grouped.append("\n\n".join(pending))
                pending = []
        if pending:
            grouped.append("\n\n".join(pending))
        for paragraph in grouped:
            if len(paragraph.split()) < 5:
                continue
            # A simple English sentence boundary, including closing quotation marks.
            boundaries = (
                [0]
                + [
                    match.end()
                    for match in re.finditer(r'[.!?]["”\x27]?\s+(?=[A-Z])', paragraph)
                ]
                + [len(paragraph)]
            )
            start, end = 0, 0
            excerpts = []
            for left, right in pairwise(boundaries):
                if end > start and len(paragraph[start:right].split()) > max_words:
                    excerpts.append(paragraph[start:end].strip())
                    start = left
                end = right
            if end > start:
                excerpts.append(paragraph[start:end].strip())
            for excerpt in excerpts:
                digest = hashlib.sha256(excerpt.encode()).hexdigest()[:10]
                chunks.append(
                    KnowledgeChunk(f"{doc.id}:{ordinal}:{digest}", doc, excerpt)
                )
                ordinal += 1
    return chunks


CHUNK_STOP_WORDS = STOP_WORDS | {
    "i",
    "my",
    "me",
    "you",
    "your",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "are",
    "was",
    "were",
    "be",
    "been",
    "can",
    "could",
    "would",
    "should",
    "will",
    "have",
    "has",
    "had",
    "with",
    "from",
    "at",
    "by",
    "as",
    "but",
    "if",
    "then",
    "do",
    "does",
    "did",
    "about",
    "into",
    "says",
    "say",
    "saying",
    "now",
    "a",
    "s",
}


def _chunk_terms(value: str) -> list[str]:
    value = value.casefold().replace("wi-fi", "wifi").replace("pop-up", "popup")
    # Normalize common English plurals for keyword matching.
    return [
        w[:-1]
        if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is"))
        else w
        for w in re.findall(r"[a-z0-9]+", value)
        if w not in CHUNK_STOP_WORDS
    ]


def search_chunks(
    query: str,
    *,
    category: str,
    expansion: str = "",
    limit: int = 3,
    per_document_limit: int = 2,
    directory: Path = DEFAULT_KNOWLEDGE_DIR,
) -> list[ChunkHit]:
    """Document relevance gate, then BM25 (k1=1.5, b=.75) over literal chunks."""
    if limit < 1:
        raise ValueError("limit must be positive")
    candidates = search_documents(
        query + " " + expansion, category=category, limit=10, directory=directory
    )
    if category == "message" and not {"apple", "icloud"}.intersection(_terms(query)):
        # A generic account message is not necessarily about an Apple Account.
        candidates = [h for h in candidates if h.document.platform == "any"]
    if not candidates:
        return []
    chunks = build_chunks([hit.document for hit in candidates])
    counts = [Counter(_chunk_terms(chunk.text)) for chunk in chunks]
    lengths = [sum(c.values()) for c in counts]
    if not chunks or not sum(lengths):
        return []
    average = sum(lengths) / len(chunks)
    primary_terms = set(_chunk_terms(query))
    expanded_terms = set(_chunk_terms(expansion)) - primary_terms
    query_terms = primary_terms | expanded_terms
    specific = query_terms - GENERIC_TERMS
    frequency = Counter(term for c in counts for term in c)
    document_scores = {h.document.id: h.score for h in candidates}
    scored = []
    for chunk, count, length in zip(chunks, counts, lengths, strict=True):
        if not specific.intersection(count):
            continue
        score = 0.0
        for term in query_terms:
            tf = count[term]
            if tf:
                idf = math.log(
                    1 + (len(chunks) - frequency[term] + 0.5) / (frequency[term] + 0.5)
                )
                weight = 1.0 if term in primary_terms else 0.2
                # Password/code evidence should not be outranked by generic
                # link words or a mistaken expanded payment signal.
                if term in primary_terms & {
                    "password",
                    "passcode",
                    "verification",
                    "credential",
                }:
                    weight *= 3
                score += (
                    weight
                    * idf
                    * tf
                    * 2.5
                    / (tf + 1.5 * (0.25 + 0.75 * length / average))
                )
        score += 0.05 * document_scores[chunk.document.id]
        scored.append(ChunkHit(chunk, round(score, 6)))
    # Cap excerpts per document so one source does not fill every result slot.
    result, per_doc = [], Counter()
    for hit in sorted(scored, key=lambda h: (-h.score, h.chunk.id)):
        if per_doc[hit.chunk.document.id] < per_document_limit:
            result.append(hit)
            per_doc[hit.chunk.document.id] += 1
        if len(result) == limit:
            break
    return result
