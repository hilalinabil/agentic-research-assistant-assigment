"""
Embedding generation component.

Sends document chunk texts to an OpenAI-compatible embeddings endpoint in
configurable batches and returns embeddings paired with their source chunks.
"""
from __future__ import annotations

import logging
import math
from typing import List, Tuple

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)


class EmbeddingValidationError(ValueError):
    """Raised when the embedding API returns an unexpected response shape."""


class Embedder:
    """
    Wraps the OpenAI embeddings API and returns validated float vectors.

    Parameters
    ----------
    client:
        A configured ``openai.OpenAI`` instance.
    model:
        The embedding model identifier (e.g. ``text-embedding-3-small``).
    batch_size:
        Maximum number of texts submitted per API call.  Defaults to 64.
    """

    def __init__(self, client: OpenAI, model: str, batch_size: int = 64) -> None:
        if not model:
            raise ValueError("Embedder: model name must not be empty.")
        if batch_size < 1:
            raise ValueError("Embedder: batch_size must be >= 1.")
        self._client = client
        self._model = model
        self._batch_size = batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of plain-text strings.

        Sends the texts to the configured model in batches, validates each
        response, and returns a flat list of float vectors in the same order
        as the input.

        Parameters
        ----------
        texts:
            Non-empty strings to embed.  Empty strings are rejected by most
            embedding APIs; callers should filter them beforehand.

        Returns
        -------
        List[List[float]]
            One embedding vector per input text, same order.

        Raises
        ------
        ValueError:
            If ``texts`` is empty.
        EmbeddingValidationError:
            If the API response length does not match the batch size, or if
            any returned embedding is not a list of floats.
        openai.APIError / openai.APIConnectionError / openai.RateLimitError:
            Propagated from the OpenAI SDK so callers can apply retry logic.
        """
        if not texts:
            raise ValueError("embed_texts: received an empty list of texts.")

        all_embeddings: List[List[float]] = []
        num_batches = math.ceil(len(texts) / self._batch_size)

        for batch_index in range(num_batches):
            start = batch_index * self._batch_size
            end = start + self._batch_size
            batch = texts[start:end]

            logger.debug(
                "Embedding batch %d/%d (%d texts) with model=%s",
                batch_index + 1,
                num_batches,
                len(batch),
                self._model,
            )

            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except RateLimitError:
                logger.error("Rate limit hit while embedding batch %d/%d.", batch_index + 1, num_batches)
                raise
            except APIConnectionError:
                logger.error("Connection error while embedding batch %d/%d.", batch_index + 1, num_batches)
                raise
            except APIError:
                logger.error("API error while embedding batch %d/%d.", batch_index + 1, num_batches)
                raise

            self._validate_response(response, expected_count=len(batch), batch_index=batch_index)

            # Sort by index field to guarantee order (the API spec allows
            # out-of-order items in the returned list).
            items = sorted(response.data, key=lambda item: item.index)
            for item in items:
                all_embeddings.append(item.embedding)

        return all_embeddings

    def embed_chunks(
        self, chunks: List[DocumentChunk]
    ) -> List[Tuple[DocumentChunk, List[float]]]:
        """
        Embed a list of :class:`~app.schemas.document.DocumentChunk` objects.

        Parameters
        ----------
        chunks:
            Chunks whose ``.text`` field will be embedded.

        Returns
        -------
        List[Tuple[DocumentChunk, List[float]]]
            Each element pairs the original chunk with its embedding vector,
            preserving input order.

        Raises
        ------
        ValueError:
            If ``chunks`` is empty.
        """
        if not chunks:
            raise ValueError("embed_chunks: received an empty list of chunks.")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        return list(zip(chunks, embeddings))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_response(response, expected_count: int, batch_index: int) -> None:
        """
        Validate that the API response contains the expected number of
        embeddings and that each is a non-empty list of finite floats.

        ``float('nan')`` and ``float('inf')`` are instances of ``float``, so
        a type-only check is insufficient.  We explicitly reject non-finite
        values because they would silently corrupt L2 normalisation and all
        subsequent similarity scores.
        """
        actual_count = len(response.data)
        if actual_count != expected_count:
            raise EmbeddingValidationError(
                f"Batch {batch_index}: expected {expected_count} embeddings, "
                f"got {actual_count}."
            )

        for i, item in enumerate(response.data):
            emb = item.embedding
            if not isinstance(emb, list) or not emb:
                raise EmbeddingValidationError(
                    f"Batch {batch_index}, item {i}: embedding is not a non-empty list."
                )
            if not all(isinstance(v, float) for v in emb):
                raise EmbeddingValidationError(
                    f"Batch {batch_index}, item {i}: embedding contains non-float values."
                )
            # Reject NaN and Inf: math.isfinite returns False for both.
            # These pass isinstance(v, float) but corrupt normalisation.
            if not all(math.isfinite(v) for v in emb):
                raise EmbeddingValidationError(
                    f"Batch {batch_index}, item {i}: embedding contains non-finite "
                    f"values (NaN or Inf)."
                )
