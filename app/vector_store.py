"""
App-level facade for Vector Store (Phase 4).
Re-exports FAISSVectorStore and helper utilities from utils.vector_store.
"""

from utils.vector_store import (
    FAISSVectorStore,
    VectorStoreError,
    get_vector_store,
)

__all__ = [
    "FAISSVectorStore",
    "VectorStoreError",
    "get_vector_store",
]
