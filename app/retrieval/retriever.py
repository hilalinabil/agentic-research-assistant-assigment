"""
Semantic retrieval component.

Accepts a natural-language query, embeds it, and returns the top-K most
similar document chunks from the in-memory vector store.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    A single ranked retrieval result.

    Attributes
    ----------
    chunk:
        The source document chunk.
    score:
        Cosine similarity in the range ``[−1, 1]``.  This is a geometric
        similarity measure, not a probability.
    """
    chunk: DocumentChunk
    score: float


class Retriever:
    """
    Query-time retrieval over a :class:`~app.retrieval.vector_store.VectorStore`.

    Design
    ------
    * The query string is embedded with the same model used at index time.
    * The query vector is L2-normalised so that cosine similarity reduces to
      a dot product against the pre-normalised stored matrix::

          scores = matrix @ query_vec   # shape (N,)

    * ``np.argsort`` with a negative score array produces a stable
      descending-score ranking that preserves insertion order for ties.

    Parameters
    ----------
    embedder:
        :class:`Embedder` configured with the embedding model.
    vector_store:
        Populated :class:`VectorStore`.
    default_top_k:
        Number of results to return when ``top_k`` is not supplied to
        :meth:`retrieve`.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("Retriever: default_top_k must be >= 1.")
        self._embedder = embedder
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Return the top-K chunks most similar to ``query``.

        Parameters
        ----------
        query:
            Non-empty natural-language question or search string.
        top_k:
            Number of results to return.  Defaults to ``self._default_top_k``.
            If fewer chunks than ``top_k`` are stored, all stored chunks are
            returned.

        Returns
        -------
        List[RetrievalResult]
            Results in descending cosine-similarity order.

        Raises
        ------
        ValueError:
            If ``query`` is empty (after stripping whitespace).
        ValueError:
            If the query embedding dimension does not match the stored
            vectors.
        """
        query = query.strip()
        if not query:
            raise ValueError("Retriever.retrieve: query must not be empty.")

        k = top_k if top_k is not None else self._default_top_k
        if k < 1:
            raise ValueError("Retriever.retrieve: top_k must be >= 1.")

        # Empty index — nothing to retrieve.
        if self._vector_store.is_empty():
            logger.debug("Retriever: vector store is empty; returning no results.")
            return []

        # Embed the query (single text, one API call).
        raw_vecs = self._embedder.embed_texts([query])
        query_vec = np.array(raw_vecs[0], dtype=np.float32)

        # Validate dimension consistency.
        stored_dim = self._vector_store.embedding_dim
        if query_vec.shape[0] != stored_dim:
            raise ValueError(
                f"Retriever: query embedding dim {query_vec.shape[0]} does not "
                f"match stored dim {stored_dim}."
            )

        # L2-normalise query vector.
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # Cosine similarity via dot product (stored vectors are pre-normalised).
        scores: np.ndarray = self._vector_store._matrix @ query_vec  # shape (N,)

        # Guard: a NaN query vector (e.g. zero-norm after normalization failure)
        # would produce NaN scores.  Detect and report rather than return
        # meaningless results.
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "Retriever: similarity scores contain non-finite values. "
                "The query embedding may be degenerate (all-zero or NaN)."
            )

        # Stable descending sort: explicitly request kind='stable' so that
        # equal-score chunks preserve their original insertion order.
        # np.argsort default is 'quicksort', which is NOT guaranteed stable.
        k_actual = min(k, self._vector_store.size)
        ranked_indices = np.argsort(-scores, kind="stable")[:k_actual]

        results: List[RetrievalResult] = []
        for idx in ranked_indices:
            results.append(
                RetrievalResult(
                    chunk=self._vector_store._chunks[idx],
                    score=float(scores[idx]),
                )
            )

        logger.debug(
            "Retriever: query='%s...', top_k=%d, returned=%d result(s).",
            query[:50],
            k,
            len(results),
        )
        return results
