"""Generator protocol."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..vectorstore.base import VectorHit


@runtime_checkable
class Generator(Protocol):
    """Compose retrieved chunks into a final answer.

    Production adapters wrap OpenAI / Anthropic / local LLMs.
    """
    name: str

    def generate(self, query: str, hits: list[VectorHit]) -> str: ...
