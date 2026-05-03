"""Embedding providers."""
from .base import EmbeddingProvider
from .simple import HashEmbedding

__all__ = ["EmbeddingProvider", "HashEmbedding"]
