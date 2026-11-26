"""
FAISS Vector Store Module for AI-Powered Smart Study Assistant (Phase 4 - RAG).
Provides persistent, per-document FAISS vector indexing, isolated semantic similarity search,
metadata synchronization, and cross-document query capabilities.
"""

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

# Optional FAISS import
_faiss_available = False
try:
    import faiss
    _faiss_available = True
except ImportError:
    faiss = None

from utils.embeddings import get_embedding_engine


class VectorStoreError(Exception):
    """Custom exception raised for vector store indexing or retrieval failures."""


class FAISSVectorStore:
    """
    Manages per-document FAISS indices with metadata persistence.
    Using per-document indices guarantees strict multi-tenant document isolation:
    queries for Document A can never accidentally match or leak chunks from Document B.
    """

    _lock = threading.Lock()

    def __init__(self, storage_dir: str | Path | None = None, dimension: int = 384):
        if storage_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            self.storage_dir = base_dir / "database" / "vector_store"
        else:
            self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension

    def _get_index_path(self, document_id: int) -> Path:
        """Returns the filesystem path for a document's binary FAISS index."""
        return self.storage_dir / f"doc_{document_id}.faiss"

    def _get_metadata_path(self, document_id: int) -> Path:
        """Returns the filesystem path for a document's chunk metadata JSON."""
        return self.storage_dir / f"doc_{document_id}_chunks.json"

    def is_document_indexed(self, document_id: int) -> bool:
        """Checks if a document already has a valid persisted FAISS index and metadata."""
        idx_path = self._get_index_path(document_id)
        meta_path = self._get_metadata_path(document_id)
        return idx_path.exists() and meta_path.exists() and idx_path.stat().st_size > 0

    def index_document(
        self,
        document_id: int,
        original_filename: str,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray | None = None,
    ) -> int:
        """
        Creates and persists a FAISS index for the provided document chunks.

        Args:
            document_id: Database ID of the document.
            original_filename: Original PDF filename for citations.
            chunks: List of chunk dicts (each containing 'chunk_text', 'page_number', etc.).
            embeddings: Optional pre-computed 2D float32 numpy array. If omitted, computed on the fly.

        Returns:
            Number of indexed chunk vectors.
        """
        if not chunks:
            return 0

        if not _faiss_available or faiss is None:
            raise VectorStoreError(
                "FAISS is not installed. Run 'pip install faiss-cpu' to enable vector search."
            )

        # Generate embeddings if not supplied
        if embeddings is None or len(embeddings) == 0:
            engine = get_embedding_engine()
            texts = [c.get("chunk_text") or c.get("text", "") for c in chunks]
            embeddings = engine.encode_batch(texts, normalize=True)

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        n_vectors, dim = embeddings.shape

        if dim != self.dimension:
            # Dynamically adjust index dimension if necessary
            self.dimension = dim

        with self._lock:
            # Create an Inner-Product index (exact cosine similarity on normalized vectors)
            index = faiss.IndexFlatIP(self.dimension)
            index.add(embeddings)

            # Persist FAISS index file
            idx_path = self._get_index_path(document_id)
            faiss.write_index(index, str(idx_path))

            # Persist metadata mapping
            meta_path = self._get_metadata_path(document_id)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

        return n_vectors

    def load_document_index(
        self, document_id: int
    ) -> tuple[Any, list[dict[str, Any]]]:
        """
        Loads the persisted FAISS index and chunk metadata for a document into memory.

        Returns:
            Tuple of (faiss_index, list_of_chunk_dicts).
        """
        if not self.is_document_indexed(document_id):
            raise VectorStoreError(
                f"Document {document_id} has not been indexed yet. Please run indexing first."
            )

        if not _faiss_available or faiss is None:
            raise VectorStoreError("FAISS is not installed.")

        idx_path = self._get_index_path(document_id)
        meta_path = self._get_metadata_path(document_id)

        try:
            index = faiss.read_index(str(idx_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            return index, chunks
        except Exception as e:
            raise VectorStoreError(
                f"Failed to load vector index for document {document_id}: {e}"
            ) from e

    def search_document(
        self,
        document_id: int,
        query: str,
        top_k: int = 4,
        threshold: float = 0.28,
        query_embedding: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """
        Executes semantic similarity search strictly within a document's chunks.

        Args:
            document_id: Document ID to search within (enforces isolation).
            query: User's question or search query string.
            top_k: Number of relevant chunks to retrieve.
            threshold: Minimum cosine similarity score threshold (0.0 to 1.0).
            query_embedding: Optional pre-computed query embedding.

        Returns:
            List of retrieved chunk dicts, each augmented with 'similarity_score' and 'rank'.
        """
        if not self.is_document_indexed(document_id):
            return []

        index, chunks = self.load_document_index(document_id)
        if index.ntotal == 0 or not chunks:
            return []

        # Encode query
        if query_embedding is None:
            engine = get_embedding_engine()
            q_emb = engine.encode_text(query, normalize=True)
        else:
            q_emb = query_embedding

        q_vec = np.ascontiguousarray(q_emb.reshape(1, -1), dtype=np.float32)

        # Search top_k nearest neighbors
        k = min(top_k, index.ntotal)
        scores, indices = index.search(q_vec, k)

        matched_chunks: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            sim_score = float(score)
            if sim_score < threshold:
                continue

            chunk_data = dict(chunks[idx])
            chunk_data["similarity_score"] = round(sim_score, 4)
            chunk_data["rank"] = rank
            matched_chunks.append(chunk_data)

        return matched_chunks

    def delete_document_index(self, document_id: int) -> bool:
        """
        Permanently removes a document's vector index and metadata files.
        """
        deleted = False
        with self._lock:
            idx_path = self._get_index_path(document_id)
            meta_path = self._get_metadata_path(document_id)
            if idx_path.exists():
                idx_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
                deleted = True
        return deleted

    def get_index_stats(self) -> dict[str, Any]:
        """Returns statistics on currently stored vector indices."""
        indices = list(self.storage_dir.glob("doc_*.faiss"))
        total_indexed_docs = len(indices)
        total_size_bytes = sum(f.stat().st_size for f in self.storage_dir.glob("*"))
        return {
            "total_indexed_documents": total_indexed_docs,
            "storage_dir": str(self.storage_dir),
            "storage_size_bytes": total_size_bytes,
        }


# Module singleton helper
_global_vector_store: FAISSVectorStore | None = None
_vs_lock = threading.Lock()


def get_vector_store(storage_dir: str | Path | None = None) -> FAISSVectorStore:
    """
    Returns a FAISSVectorStore instance.

    Resolution order (first match wins):
    1. ``storage_dir`` argument explicitly provided by the caller.
    2. Flask ``current_app.config["VECTOR_STORE_DIR"]`` when running inside an app context.
    3. Module-level singleton pointing to the default ``database/vector_store/`` directory.

    Notes:
        - When Flask app config supplies a custom dir (e.g. in tests), a *fresh*
          instance is returned rather than the cached singleton, so that each test
          can use its own isolated directory without contaminating the global state.
        - Outside Flask (scripts, CLI), the singleton behaviour is unchanged.
    """
    global _global_vector_store

    # Prefer explicit argument, then Flask app config
    resolved_dir: str | Path | None = storage_dir
    if resolved_dir is None:
        try:
            from flask import current_app
            resolved_dir = current_app.config.get("VECTOR_STORE_DIR")
        except RuntimeError:
            # Not inside a Flask application context — normal for scripts / tests that
            # exercise utilities directly without Flask.
            resolved_dir = None

    if resolved_dir is not None:
        # A custom directory was requested — check if the global singleton already
        # points to the same directory.  If not, return a per-call instance so we
        # never silently share state across tests or environments.
        resolved_path = Path(resolved_dir).resolve()
        if _global_vector_store is not None:
            existing_path = Path(_global_vector_store.storage_dir).resolve()
            if existing_path != resolved_path:
                # Different directory — return a fresh isolated instance
                return FAISSVectorStore(storage_dir=resolved_path)
        # Either singleton not yet created, or same dir — fall through to singleton init
        if _global_vector_store is None:
            with _vs_lock:
                if _global_vector_store is None:
                    _global_vector_store = FAISSVectorStore(storage_dir=resolved_path)
        return _global_vector_store

    # Default path: use (or create) the module-level singleton
    if _global_vector_store is None:
        with _vs_lock:
            if _global_vector_store is None:
                _global_vector_store = FAISSVectorStore(storage_dir=None)
    return _global_vector_store


def reset_vector_store() -> None:
    """
    Resets the module-level singleton to ``None``.
    Intended for use in test teardown only — do not call in production code.
    """
    global _global_vector_store
    with _vs_lock:
        _global_vector_store = None
