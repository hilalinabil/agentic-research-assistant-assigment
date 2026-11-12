"""app.retrieval — Embedding, vector storage, and semantic retrieval."""

from app.retrieval.embedder import Embedder, EmbeddingValidationError
from app.retrieval.retriever import RetrievalResult, Retriever
from app.retrieval.vector_store import DimensionMismatchError, VectorStore

__all__ = [
    "Embedder",
    "EmbeddingValidationError",
    "VectorStore",
    "DimensionMismatchError",
    "Retriever",
    "RetrievalResult",
]
