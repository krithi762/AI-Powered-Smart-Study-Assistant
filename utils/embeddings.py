"""
Embedding Generation Module for AI-Powered Smart Study Assistant (Phase 4 - RAG).
Provides thread-safe singleton loading for SentenceTransformers ('all-MiniLM-L6-v2'),
vector normalization, batch embedding generation, and cosine similarity utilities.
"""

import os
import threading
from typing import Any, Optional

import numpy as np

# Point HuggingFace cache to D: drive
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_hf_cache = os.path.join(_base_dir, "models", "cache")
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Lazy import flag
_sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    _sentence_transformers_available = True
except ImportError:
    SentenceTransformer = None


class EmbeddingError(Exception):
    """Custom exception raised for embedding generation failures."""


class EmbeddingEngine:
    """
    Thread-safe Singleton wrapper for SentenceTransformer embedding generation.
    Generates normalized 384-dimensional dense semantic vectors using 'all-MiniLM-L6-v2'.
    """

    _instance: Optional["EmbeddingEngine"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._load_lock = threading.Lock()
        self._dimension = 384

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingEngine":
        """Returns the shared singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name=model_name)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Checks if the embedding model weights are loaded into memory."""
        return self._model is not None

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimensionality (384 for all-MiniLM-L6-v2)."""
        return self._dimension

    def load_model(self) -> None:
        """Loads the SentenceTransformer model if not already loaded."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            if not _sentence_transformers_available or SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed in the active environment. "
                    "Run 'pip install sentence-transformers'."
                )

            # Load model (cached in models/cache)
            self._model = SentenceTransformer(self.model_name)
            # Infer dimension cleanly
            try:
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension() or 384
                elif hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dimension = self._model.get_sentence_embedding_dimension() or 384
                else:
                    self._dimension = 384
            except (AttributeError, TypeError, ValueError):
                self._dimension = 384

    def encode_text(
        self, text: str, normalize: bool = True
    ) -> np.ndarray:
        """
        Generates an embedding vector for a single text query or snippet.

        Args:
            text: Input string.
            normalize: Whether to L2-normalize the vector (enables exact dot-product cosine similarity).

        Returns:
            1D float32 numpy array with shape (dimension,).
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self._dimension, dtype=np.float32)

        self.load_model()
        if self._model is None:
            raise EmbeddingError("Embedding model is not loaded.")
        emb = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return emb.astype(np.float32)

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Generates embedding vectors for a batch of text chunks.

        Args:
            texts: List of text strings.
            batch_size: Processing batch size.
            normalize: Whether to L2-normalize embeddings.
            show_progress_bar: Whether to display a progress indicator during encoding.

        Returns:
            2D float32 numpy array with shape (len(texts), dimension).
        """
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)

        self.load_model()
        if self._model is None:
            raise EmbeddingError("Embedding model is not loaded.")
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embs.astype(np.float32)


# Module-level convenience functions
def get_embedding_engine(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingEngine:
    """Helper to access the singleton EmbeddingEngine."""
    return EmbeddingEngine.get_instance(model_name=model_name)


def compute_text_embedding(text: str) -> np.ndarray:
    """Generates an L2-normalized embedding for a query string."""
    engine = get_embedding_engine()
    return engine.encode_text(text, normalize=True)


def compute_chunks_embeddings(chunks: list[dict[str, Any]]) -> np.ndarray:
    """Extracts text from a list of chunk dicts and generates their batch embeddings."""
    texts = [c.get("chunk_text") or c.get("text", "") for c in chunks]
    engine = get_embedding_engine()
    return engine.encode_batch(texts, normalize=True)
