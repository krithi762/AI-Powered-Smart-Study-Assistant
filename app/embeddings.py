"""
App-level facade for Embedding Generation (Phase 4).
Re-exports EmbeddingEngine and helper utilities from utils.embeddings.
"""

from utils.embeddings import (
    EmbeddingEngine,
    compute_chunks_embeddings,
    compute_text_embedding,
    get_embedding_engine,
)

__all__ = [
    "EmbeddingEngine",
    "compute_chunks_embeddings",
    "compute_text_embedding",
    "get_embedding_engine",
]
