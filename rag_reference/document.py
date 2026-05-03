"""Document + Chunk dataclasses + token-aware chunking."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A whole document about to be ingested."""
    doc_id:   str
    text:     str
    source:   str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("Document.doc_id required")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Document.text must be a non-empty string")


@dataclass
class Chunk:
    """A windowed slice of a document."""
    chunk_id: str
    doc_id:   str
    text:     str
    start:    int = 0
    end:      int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start


def _tokenize(text: str) -> list[str]:
    """Whitespace+punct word tokenizer; cheap, good enough for chunk sizing."""
    return re.findall(r"\S+", text)


def chunk_document(doc: Document, *,
                   chunk_size: int = 200,
                   overlap: int = 40) -> list[Chunk]:
    """Split `doc.text` into overlapping windows of ~`chunk_size` words.

    `overlap` words carry over between adjacent chunks so a sentence
    spanning the boundary is preserved in at least one chunk.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    words = _tokenize(doc.text)
    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    i = 0
    chunk_idx = 0
    while i < len(words):
        window = words[i:i + chunk_size]
        if not window:
            break
        text = " ".join(window)
        cid = _stable_chunk_id(doc.doc_id, chunk_idx, text)
        chunks.append(Chunk(
            chunk_id=cid,
            doc_id=doc.doc_id,
            text=text,
            start=i,
            end=i + len(window),
            metadata=dict(doc.metadata),
        ))
        chunk_idx += 1
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks


def _stable_chunk_id(doc_id: str, idx: int, text: str) -> str:
    """Deterministic id so re-chunking the same doc yields the same ids."""
    h = hashlib.sha1(f"{doc_id}|{idx}|{text}".encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}-{idx:04d}-{h}"
