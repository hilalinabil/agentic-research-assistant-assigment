"""
In-memory vector store backed by NumPy.

Stores L2-normalised embedding vectors alongside their source
:class:`~app.schemas.document.DocumentChunk` objects.  Cosine similarity
is then computed as a simple dot product, avoiding per-query division.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)


class DimensionMismatchError(ValueError):
    """
    Raised when an embedding vector has a dimension that conflicts with the
    vectors already stored in the index.
    """


class VectorStore:
    """
    Thread-unsafe, in-memory vector index.

    Embedding vectors are L2-normalised on insertion so that cosine
    similarity between any two stored vectors reduces to their dot product.
    This design trades a small amount of storage accuracy for simpler and
    faster similarity queries.

    Attributes
    ----------
    _matrix : np.ndarray | None
        Shape ``(N, D)``, dtype ``float32``.  ``None`` until the first
        :meth:`add` call.
    _chunks : List[DocumentChunk]
        Parallel list whose ``i``-th element corresponds to row ``i`` of
        ``_matrix``.
    _chunk_id_to_index : Dict[str, int]
        Fast lookup from ``chunk_id`` to row index.
    """

    def __init__(self) -> None:
        self._matrix: Optional[np.ndarray] = None   # shape (N, D), float32
        self._chunks: List[DocumentChunk] = []
        self._chunk_id_to_index: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return ``True`` if no vectors have been added yet."""
        return self._matrix is None

    @property
    def embedding_dim(self) -> int:
        """
        Dimensionality of stored vectors.

        Raises
        ------
        RuntimeError
            If the store is empty.
        """
        if self._matrix is None:
            raise RuntimeError("VectorStore is empty; embedding_dim is not defined.")
        return self._matrix.shape[1]

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> None:
        """
        Insert chunks and their corresponding embedding vectors.

        All embeddings must have the same dimensionality.  If the store
        already contains vectors, the new embeddings must match the existing
        dimension.

        Parameters
        ----------
        chunks:
            Source document chunks.
        embeddings:
            One raw (un-normalised) embedding vector per chunk.  Vectors are
            L2-normalised before storage.

        Raises
        ------
        ValueError:
            If ``chunks`` and ``embeddings`` lengths do not match, or if
            ``chunks`` is empty.
        DimensionMismatchError:
            If any embedding has a dimension that differs from others in
            this batch, or from vectors already stored.
        """
        if not chunks:
            raise ValueError("VectorStore.add: chunks list must not be empty.")
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"VectorStore.add: chunks ({len(chunks)}) and embeddings "
                f"({len(embeddings)}) must have equal length."
            )

        # Convert to float32 matrix for this batch.
        try:
            batch_matrix = np.array(embeddings, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"VectorStore.add: could not convert embeddings to numpy array: {exc}") from exc

        if batch_matrix.ndim != 2:
            raise ValueError("VectorStore.add: each embedding must be a 1-D float vector.")

        batch_dim = batch_matrix.shape[1]

        # Check dimensional consistency within the batch.
        # (np.array would have already raised if rows differ, but be explicit.)
        if batch_matrix.shape[0] > 0 and batch_dim == 0:
            raise DimensionMismatchError("VectorStore.add: embedding dimension must be > 0.")

        # Check against existing stored dimension.
        if self._matrix is not None and batch_dim != self._matrix.shape[1]:
            raise DimensionMismatchError(
                f"VectorStore.add: batch embedding dim {batch_dim} does not match "
                f"stored dim {self._matrix.shape[1]}."
            )

        # L2-normalise — vectors with zero norm become zero vectors.
        norms = np.linalg.norm(batch_matrix, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors (degenerate case).
        # A zero-norm vector stays as all-zeros after normalisation, which is
        # mathematically correct (zero similarity with everything).
        norms = np.where(norms == 0, 1.0, norms)
        normalised = (batch_matrix / norms).astype(np.float32)

        # Guard: reject NaN or Inf values in the raw input.  NaN propagates
        # through division, so a single bad value would silently poison every
        # score involving that vector.  We check the raw matrix rather than
        # the normalised result so the error message is more actionable.
        if not np.all(np.isfinite(batch_matrix)):
            raise ValueError(
                "VectorStore.add: one or more embeddings contain non-finite "
                "values (NaN or Inf).  Inspect the embedding source."
            )

        # Append to the matrix.
        base_index = len(self._chunks)
        if self._matrix is None:
            self._matrix = normalised
        else:
            self._matrix = np.concatenate([self._matrix, normalised], axis=0)

        # Update parallel chunk list and id→index mapping.
        for offset, chunk in enumerate(chunks):
            self._chunk_id_to_index[chunk.chunk_id] = base_index + offset
            self._chunks.append(chunk)

        logger.debug(
            "VectorStore: added %d vectors (total=%d, dim=%d).",
            len(chunks),
            self.size,
            batch_dim,
        )
