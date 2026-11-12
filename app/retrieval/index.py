"""
Builds the in-memory search index at application startup.

The corpus is small (~21 documents, ~500 chunks), so the index is rebuilt
from scratch each time the process starts: load -> chunk -> embed -> store.
This avoids a persistent cache that could silently go stale when the corpus,
chunking parameters or embedding model change.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

from app.core.config import Settings
from app.ingestion.chunker import MarkdownSectionChunker
from app.ingestion.loader import CorpusLoader
from app.retrieval.embedder import Embedder
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.document import Document, DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class CorpusIndex:
    """Everything the agent needs at query time."""
    documents: List[Document]
    chunks: List[DocumentChunk]
    retriever: Retriever

    @property
    def filenames(self) -> List[str]:
        return [d.filename for d in self.documents]

    def drug_coverage(self, drugs: List[str]) -> Dict[str, List[str]]:
        """Map each drug name to the corpus files dedicated to it (empty list = no dedicated document).

        Uses the drug name the loader derives from each filename (e.g. ``dailymed-warfarin.md``).
        A drug without a dedicated document may still be *mentioned* inside other documents.
        """
        out: Dict[str, List[str]] = {}
        for drug in drugs:
            name = drug.strip().lower()
            out[drug] = sorted(d.filename for d in self.documents
                               if d.drug_name and d.drug_name.lower() in name)
        return out


def build_index(settings: Settings, embedder: Embedder) -> CorpusIndex:
    """Load, chunk and embed the corpus into an in-memory vector store."""
    started = time.perf_counter()
    documents = CorpusLoader(settings.corpus_dir).load_all()
    if not documents:
        raise RuntimeError(f"No documents found in corpus directory {settings.corpus_dir}")

    chunker = MarkdownSectionChunker(
        target_chunk_size=settings.target_chunk_size,
        overlap_tokens=settings.overlap_tokens,
    )
    chunks = chunker.chunk_all(documents)

    pairs = embedder.embed_chunks(chunks)
    store = VectorStore()
    store.add([c for c, _ in pairs], [v for _, v in pairs])

    retriever = Retriever(embedder=embedder, vector_store=store, default_top_k=settings.default_top_k)
    logger.info(
        "Index ready: %d documents, %d chunks, dim=%d (%.1fs)",
        len(documents), len(chunks), store.embedding_dim, time.perf_counter() - started,
    )
    return CorpusIndex(documents=documents, chunks=chunks, retriever=retriever)
