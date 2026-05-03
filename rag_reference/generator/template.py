"""TemplateGenerator — dependency-free deterministic generator.

Renders the retrieved hits as a numbered citation block followed by
a stitched answer. Useful for tests and as a "no-LLM" baseline that
proves the rest of the pipeline (chunk → embed → retrieve → rerank)
is wired correctly.

Production stacks replace this with an LLM-backed generator, while
keeping the same Generator protocol.
"""
from __future__ import annotations

from ..vectorstore.base import VectorHit


class TemplateGenerator:
    """Renders retrieved chunks into a citation-anchored answer."""
    name = "template_generator"

    def __init__(self, *, max_chunks: int = 5,
                 max_chars_per_chunk: int = 300):
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk

    def generate(self, query: str, hits: list[VectorHit]) -> str:
        if not hits:
            return f"Q: {query}\nA: (no relevant context retrieved)"
        sel = hits[: self.max_chunks]

        lines = [f"Q: {query}", ""]
        lines.append("Based on the retrieved context:")
        lines.append("")
        for i, h in enumerate(sel, start=1):
            snippet = h.text.strip()
            if len(snippet) > self.max_chars_per_chunk:
                snippet = snippet[: self.max_chars_per_chunk - 1] + "…"
            lines.append(f"  [{i}] {snippet}")
        lines.append("")
        lines.append("Citations:")
        for i, h in enumerate(sel, start=1):
            lines.append(f"  [{i}] chunk_id={h.chunk_id} score={h.score:.4f}")
        return "\n".join(lines)
