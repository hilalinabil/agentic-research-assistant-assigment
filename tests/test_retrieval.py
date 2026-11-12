"""
Phase 2 — Retrieval unit tests.

All tests that touch the OpenAI API are fully mocked.
No real HTTP calls are made during test execution.

Test inventory
--------------
Similarity / normalization
  test_cosine_similarity_identical        identical vectors → score ≈ 1.0
  test_cosine_similarity_orthogonal       orthogonal vectors → score ≈ 0.0
  test_cosine_similarity_opposite         opposite vectors → score ≈ -1.0
  test_l2_normalization                   stored vectors have unit L2-norm

Ranking / top-k
  test_ranking_order                      results returned highest-score first
  test_top_k_fewer_than_k                 N < k → returns all N
  test_top_k_exact                        N >= k → returns exactly k

Edge cases
  test_empty_index_returns_empty          retrieve() on empty store → []
  test_empty_query_raises                 empty / whitespace-only → ValueError
  test_dimension_mismatch_add_raises      mismatched dim on add → error
  test_dimension_mismatch_query_raises    mismatched query dim → ValueError

Metadata / identity
  test_metadata_preserved                 retrieved chunk has correct ids/fields

Error propagation
  test_embedding_api_failure_propagates   APIError from embedder propagates

Batching
  test_embedding_batch_splitting          70 chunks with batch_size=64 → 2 calls

VectorStore internals
  test_vector_store_empty_at_start
  test_vector_store_size_after_add
  test_vector_store_chunk_id_lookup
  test_add_mismatched_lengths_raises
  test_add_empty_chunks_raises

Embedder internals
  test_embed_texts_empty_raises
  test_embed_chunks_empty_raises
  test_embedder_validates_response_length
  test_embedder_response_sorted_by_index

Integration (skipped in CI)
  test_corpus_integration                 full load+embed+retrieve round-trip
"""
from __future__ import annotations

import math
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

from openai import APIError

from app.retrieval.embedder import Embedder, EmbeddingValidationError
from app.retrieval.vector_store import VectorStore, DimensionMismatchError
from app.retrieval.retriever import Retriever, RetrievalResult
from app.schemas.document import DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4  # small dimension for fast unit tests


def _make_chunk(chunk_id: str = "chunk-001", source: str = "drug.md") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_filename=source,
        section_title="H2 Section",
        section_path="H1 > H2",
        text="Sample pharmacology text.",
        token_count=5,
        start_line=1,
        end_line=3,
        part_index=0,
        total_parts=1,
    )


def _make_chunks(n: int) -> list[DocumentChunk]:
    return [_make_chunk(chunk_id=f"chunk-{i:03d}") for i in range(n)]


def _mock_embedding_response(vectors: list[list[float]]) -> MagicMock:
    """
    Build a mock that mimics openai.types.CreateEmbeddingResponse.
    Items are returned in-order and include an ``.index`` attribute.
    """
    response = MagicMock()
    items = []
    for idx, vec in enumerate(vectors):
        item = MagicMock()
        item.index = idx
        item.embedding = [float(v) for v in vec]
        items.append(item)
    response.data = items
    return response


def _unit_vec(dim: int, pos: int = 0) -> list[float]:
    """Return a unit vector of length ``dim`` with a 1.0 at position ``pos``."""
    v = [0.0] * dim
    v[pos] = 1.0
    return v


def _make_embedder_with_mock(vectors: list[list[float]], batch_size: int = 64) -> tuple[Embedder, MagicMock]:
    """Return an Embedder whose underlying client is mocked to return ``vectors``."""
    mock_client = MagicMock()
    embedder = Embedder(client=mock_client, model="test-model", batch_size=batch_size)
    mock_client.embeddings.create.return_value = _mock_embedding_response(vectors)
    return embedder, mock_client


# ---------------------------------------------------------------------------
# VectorStore — internals
# ---------------------------------------------------------------------------

class TestVectorStoreInternals:

    def test_vector_store_empty_at_start(self):
        store = VectorStore()
        assert store.is_empty()
        assert store.size == 0

    def test_vector_store_size_after_add(self):
        store = VectorStore()
        chunks = _make_chunks(3)
        embeddings = [_unit_vec(DIM, i % DIM) for i in range(3)]
        store.add(chunks, embeddings)
        assert store.size == 3
        assert not store.is_empty()

    def test_vector_store_chunk_id_lookup(self):
        store = VectorStore()
        chunk = _make_chunk(chunk_id="drug-001")
        store.add([chunk], [_unit_vec(DIM)])
        assert "drug-001" in store._chunk_id_to_index
        idx = store._chunk_id_to_index["drug-001"]
        assert store._chunks[idx].chunk_id == "drug-001"

    def test_add_mismatched_lengths_raises(self):
        store = VectorStore()
        chunks = _make_chunks(2)
        embeddings = [_unit_vec(DIM)]  # only 1 embedding for 2 chunks
        with pytest.raises(ValueError, match="equal length"):
            store.add(chunks, embeddings)

    def test_add_empty_chunks_raises(self):
        store = VectorStore()
        with pytest.raises(ValueError, match="must not be empty"):
            store.add([], [])


# ---------------------------------------------------------------------------
# Similarity geometry
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """
    Insert two chunks and retrieve against a known query to verify scores.
    The embedder is mocked; we insert known unit vectors directly.
    """

    def _store_with_vecs(self, vecs: list[list[float]]) -> tuple[VectorStore, list[DocumentChunk]]:
        store = VectorStore()
        chunks = _make_chunks(len(vecs))
        store.add(chunks, vecs)
        return store, chunks

    def _retriever(self, store: VectorStore, query_vec: list[float]) -> list[RetrievalResult]:
        embedder, mock_client = _make_embedder_with_mock([query_vec])
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=10)
        return retriever.retrieve("test query")

    def test_cosine_similarity_identical(self):
        vec = _unit_vec(DIM, 0)
        store, _ = self._store_with_vecs([vec])
        results = self._retriever(store, vec)
        assert len(results) == 1
        assert abs(results[0].score - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        store, _ = self._store_with_vecs([_unit_vec(DIM, 0)])
        results = self._retriever(store, _unit_vec(DIM, 1))
        assert len(results) == 1
        assert abs(results[0].score - 0.0) < 1e-5

    def test_cosine_similarity_opposite(self):
        vec = _unit_vec(DIM, 0)
        opposite = [-v for v in vec]
        store, _ = self._store_with_vecs([vec])
        results = self._retriever(store, opposite)
        assert len(results) == 1
        assert abs(results[0].score - (-1.0)) < 1e-5

    def test_l2_normalization(self):
        """Vectors stored in the matrix must all have unit L2-norm."""
        # Insert un-normalised vectors; store must normalise them.
        raw_vecs = [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 5.0, 12.0]]
        store = VectorStore()
        store.add(_make_chunks(2), raw_vecs)
        norms = np.linalg.norm(store._matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Ranking / top-k
# ---------------------------------------------------------------------------

class TestRankingAndTopK:

    def _build_retriever(self, stored_vecs, query_vec, top_k=5):
        store = VectorStore()
        chunks = _make_chunks(len(stored_vecs))
        store.add(chunks, stored_vecs)
        embedder, _ = _make_embedder_with_mock([query_vec])
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=top_k)
        return retriever

    def test_ranking_order(self):
        """Results must be returned in strictly descending similarity order."""
        # vec at pos-0 is most similar to query at pos-0
        stored = [_unit_vec(DIM, 0), _unit_vec(DIM, 1), _unit_vec(DIM, 2)]
        query = _unit_vec(DIM, 0)
        retriever = self._build_retriever(stored, query, top_k=3)
        results = retriever.retrieve("q")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results are not in descending order"
        assert abs(results[0].score - 1.0) < 1e-5

    def test_top_k_fewer_than_k(self):
        """When N < k, all N chunks are returned."""
        stored = [_unit_vec(DIM, 0), _unit_vec(DIM, 1)]
        retriever = self._build_retriever(stored, _unit_vec(DIM, 0), top_k=10)
        results = retriever.retrieve("q")
        assert len(results) == 2

    def test_top_k_exact(self):
        """When N >= k, exactly k results are returned."""
        stored = [_unit_vec(DIM, i % DIM) for i in range(8)]
        retriever = self._build_retriever(stored, _unit_vec(DIM, 0), top_k=3)
        results = retriever.retrieve("q")
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_index_returns_empty(self):
        store = VectorStore()
        embedder, _ = _make_embedder_with_mock([_unit_vec(DIM)])
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=5)
        results = retriever.retrieve("any query")
        assert results == []

    def test_empty_query_raises(self):
        store = VectorStore()
        embedder, _ = _make_embedder_with_mock([_unit_vec(DIM)])
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=5)
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("")

    def test_whitespace_only_query_raises(self):
        store = VectorStore()
        embedder, _ = _make_embedder_with_mock([_unit_vec(DIM)])
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=5)
        with pytest.raises(ValueError, match="empty"):
            retriever.retrieve("   ")

    def test_dimension_mismatch_add_raises(self):
        """Adding a second batch with wrong dimension raises DimensionMismatchError."""
        store = VectorStore()
        store.add(_make_chunks(1), [_unit_vec(DIM)])
        with pytest.raises(DimensionMismatchError):
            store.add(_make_chunks(1), [_unit_vec(DIM + 1)])  # wrong dim

    def test_dimension_mismatch_query_raises(self):
        """Query vector with wrong dimension raises ValueError."""
        store = VectorStore()
        store.add(_make_chunks(1), [_unit_vec(DIM)])
        # Return a query vector with the wrong dim
        embedder, _ = _make_embedder_with_mock([_unit_vec(DIM + 1)])
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=5)
        with pytest.raises(ValueError, match="dim"):
            retriever.retrieve("query")


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------

class TestMetadataPreserved:

    def test_metadata_preserved(self):
        """Retrieved chunk must retain its original chunk_id, source_filename, section_path."""
        chunk = DocumentChunk(
            chunk_id="ibuprofen-001",
            source_filename="dailymed-ibuprofen.md",
            section_title="Contraindications",
            section_path="Ibuprofen > Contraindications",
            text="Do not use if allergic.",
            token_count=6,
            start_line=10,
            end_line=12,
            part_index=0,
            total_parts=1,
        )

        store = VectorStore()
        store.add([chunk], [_unit_vec(DIM)])

        embedder, _ = _make_embedder_with_mock([_unit_vec(DIM)])
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=1)
        results = retriever.retrieve("ibuprofen allergy")

        assert len(results) == 1
        rc = results[0].chunk
        assert rc.chunk_id == "ibuprofen-001"
        assert rc.source_filename == "dailymed-ibuprofen.md"
        assert rc.section_path == "Ibuprofen > Contraindications"


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

class TestErrorPropagation:

    def test_embedding_api_failure_propagates(self):
        """An APIError raised by the embedder must propagate from retriever.retrieve()."""
        store = VectorStore()
        store.add(_make_chunks(1), [_unit_vec(DIM)])

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = APIError(
            message="server error", request=MagicMock(), body=None
        )
        embedder = Embedder(client=mock_client, model="test-model")
        retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=5)

        with pytest.raises(APIError):
            retriever.retrieve("warfarin")


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class TestBatching:

    def test_embedding_batch_splitting(self):
        """
        70 chunks with batch_size=64 must trigger exactly 2 API calls:
        one for the first 64 and one for the remaining 6.
        """
        n_chunks = 70
        batch_size = 64
        dim = DIM

        chunks = _make_chunks(n_chunks)
        texts = [c.text for c in chunks]

        mock_client = MagicMock()

        def side_effect(model, input):  # noqa: A002
            # Return one mock embedding per input item.
            return _mock_embedding_response([[0.1] * dim] * len(input))

        mock_client.embeddings.create.side_effect = side_effect

        embedder = Embedder(client=mock_client, model="test-model", batch_size=batch_size)
        embedder.embed_texts(texts)

        expected_calls = math.ceil(n_chunks / batch_size)
        assert mock_client.embeddings.create.call_count == expected_calls

        # Verify batch sizes: first call gets 64, second gets 6.
        actual_calls = mock_client.embeddings.create.call_args_list
        assert len(actual_calls[0].kwargs["input"]) == 64
        assert len(actual_calls[1].kwargs["input"]) == 6


# ---------------------------------------------------------------------------
# Embedder internals
# ---------------------------------------------------------------------------

class TestEmbedderInternals:

    def test_embed_texts_empty_raises(self):
        mock_client = MagicMock()
        embedder = Embedder(client=mock_client, model="m")
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_texts([])

    def test_embed_chunks_empty_raises(self):
        mock_client = MagicMock()
        embedder = Embedder(client=mock_client, model="m")
        with pytest.raises(ValueError, match="empty"):
            embedder.embed_chunks([])

    def test_embedder_validates_response_length(self):
        """If the API returns fewer embeddings than requested, raise EmbeddingValidationError."""
        mock_client = MagicMock()
        # Request 2 texts but API returns only 1 embedding.
        mock_client.embeddings.create.return_value = _mock_embedding_response([[0.1] * DIM])
        embedder = Embedder(client=mock_client, model="m")
        with pytest.raises(EmbeddingValidationError, match="expected 2"):
            embedder.embed_texts(["a", "b"])

    def test_embedder_response_sorted_by_index(self):
        """
        If the API returns items out of order (item.index not sequential),
        the embedder must sort by index so the output matches input order.
        """
        mock_client = MagicMock()
        # Deliberately return items in reverse index order.
        item0 = MagicMock()
        item0.index = 1
        item0.embedding = [1.0] * DIM  # This should map to texts[1]

        item1 = MagicMock()
        item1.index = 0
        item1.embedding = [0.0] * DIM  # This should map to texts[0]

        response = MagicMock()
        response.data = [item0, item1]
        mock_client.embeddings.create.return_value = response

        embedder = Embedder(client=mock_client, model="m")
        results = embedder.embed_texts(["first", "second"])

        # After sort-by-index, index=0 → [0.0]*DIM, index=1 → [1.0]*DIM
        assert results[0] == [0.0] * DIM
        assert results[1] == [1.0] * DIM


# ---------------------------------------------------------------------------
# Integration smoke test (skipped in CI — requires real credentials)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_corpus_integration():
    """
    Full round-trip: load corpus → chunk → embed → retrieve.

    Requires OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_EMBEDDING_MODEL in .env.
    Run manually with:
        RUN_INTEGRATION=1 .venv/bin/pytest -v tests/test_retrieval.py::test_corpus_integration -s
    """
    from app.core.config import settings
    from app.core.llm_client import get_openai_client
    from app.ingestion.loader import CorpusLoader
    from app.ingestion.chunker import MarkdownSectionChunker

    loader = CorpusLoader(corpus_dir=settings.corpus_dir)
    docs = loader.load_all()
    assert len(docs) > 0, "No documents loaded from corpus."

    chunker = MarkdownSectionChunker()
    chunks = chunker.chunk_all(docs)
    assert len(chunks) > 0, "No chunks produced."

    client = get_openai_client()
    embedder = Embedder(
        client=client,
        model=settings.openai_embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    # Embed only a small sample to keep test fast.
    sample = chunks[:10]
    pairs = embedder.embed_chunks(sample)
    assert len(pairs) == 10

    dim = len(pairs[0][1])
    assert dim > 0, "Embedding dimension must be > 0."

    store = VectorStore()
    store.add([p[0] for p in pairs], [p[1] for p in pairs])
    assert store.size == 10
    assert store.embedding_dim == dim

    retriever = Retriever(
        embedder=embedder,
        vector_store=store,
        default_top_k=3,
    )
    results = retriever.retrieve("warfarin drug interactions")
    assert 1 <= len(results) <= 3
    for r in results:
        assert -1.0 <= r.score <= 1.0
        assert r.chunk.chunk_id
        assert r.chunk.source_filename
    print(f"\nIntegration test passed. dim={dim}, top result: {results[0].chunk.source_filename} score={results[0].score:.4f}")
