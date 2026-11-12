"""
Unit and integration tests for Phase 1: Document Ingestion and Section-Aware Chunking.
"""
from pathlib import Path
import pytest

from app.schemas.document import Document, DocumentChunk
from app.ingestion.loader import CorpusLoader
from app.ingestion.chunker import MarkdownSectionChunker, count_tokens


# ---------------------------------------------------------------------------
# CorpusLoader Tests
# ---------------------------------------------------------------------------

def test_loader_missing_directory(tmp_path: Path):
    """CorpusLoader must raise FileNotFoundError if directory does not exist."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        CorpusLoader(non_existent)


def test_loader_markdown_and_text(tmp_path: Path):
    """CorpusLoader must load both .md and .txt files."""
    md_file = tmp_path / "doc1.md"
    md_file.write_text("# Markdown Doc\n\nContent for markdown.", encoding="utf-8")

    txt_file = tmp_path / "doc2.txt"
    txt_file.write_text("Text Doc\n\nContent for plain text.", encoding="utf-8")

    loader = CorpusLoader(tmp_path)
    docs = loader.load_all()

    assert len(docs) == 2
    filenames = {d.filename for d in docs}
    assert "doc1.md" in filenames
    assert "doc2.txt" in filenames


def test_loader_unsupported_extensions(tmp_path: Path):
    """CorpusLoader must ignore unsupported file extensions (.csv, .json, .py, etc.)."""
    (tmp_path / "valid.md").write_text("# Valid\n\nValid markdown.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("col1,col2\nval1,val2", encoding="utf-8")
    (tmp_path / "ignored.json").write_text('{"key": "value"}', encoding="utf-8")
    (tmp_path / "ignored.py").write_text('print("hello")', encoding="utf-8")

    loader = CorpusLoader(tmp_path)
    discovered = loader.discover_files()

    assert len(discovered) == 1
    assert discovered[0].name == "valid.md"


def test_loader_empty_files(tmp_path: Path):
    """CorpusLoader must skip zero-byte empty files without crashing."""
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "valid.md").write_text("# Valid\n\nValid content.", encoding="utf-8")

    loader = CorpusLoader(tmp_path)
    docs = loader.load_all()

    assert len(docs) == 1
    assert docs[0].filename == "valid.md"


def test_loader_encoding_fallback(tmp_path: Path):
    """CorpusLoader must handle non-UTF-8 encodings gracefully and record explicit warnings."""
    latin1_file = tmp_path / "latin1_doc.md"
    # Write byte 0xE9 (é in Latin-1, invalid standalone byte in UTF-8)
    latin1_file.write_bytes(b"# Latin1 Header\n\nCaf\xe9 and na\xefve.\n")

    loader = CorpusLoader(tmp_path)
    doc = loader.load_document(latin1_file)

    assert doc is not None
    assert "Café" in doc.content or "Caf" in doc.content
    assert doc.metadata["encoding"] == "latin-1 (fallback)"
    assert "encoding_warning" in doc.metadata
    assert len(loader.encoding_warnings) == 1
    assert "latin-1 fallback" in loader.encoding_warnings[0]


def test_loader_metadata_classification(tmp_path: Path):
    """CorpusLoader must classify document types and extract drug names from conventions."""
    (tmp_path / "dailymed-warfarin.md").write_text("# Warfarin Label\n\nText", encoding="utf-8")
    (tmp_path / "medlineplus-apixaban.md").write_text("# Apixaban Guide\n\nText", encoding="utf-8")
    (tmp_path / "GLOSSARY.md").write_text("# Glossary\n\nText", encoding="utf-8")

    loader = CorpusLoader(tmp_path)
    docs = {d.filename: d for d in loader.load_all()}

    assert docs["dailymed-warfarin.md"].source_type == "DailyMed"
    assert docs["dailymed-warfarin.md"].drug_name == "Warfarin"

    assert docs["medlineplus-apixaban.md"].source_type == "MedlinePlus"
    assert docs["medlineplus-apixaban.md"].drug_name == "Apixaban"

    assert docs["GLOSSARY.md"].source_type == "Glossary"
    assert docs["GLOSSARY.md"].drug_name is None


def test_loader_path_traversal_confinement(tmp_path: Path):
    """CorpusLoader must prevent loading files outside its designated directory."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    outside_file = tmp_path / "secret.md"
    outside_file.write_text("# Secret\n\nPrivate data", encoding="utf-8")

    loader = CorpusLoader(corpus_dir)
    # Attempting to explicitly pass an outside file to load_document must return None
    loaded = loader.load_document(outside_file)
    assert loaded is None


# ---------------------------------------------------------------------------
# MarkdownSectionChunker Tests
# ---------------------------------------------------------------------------

def test_chunker_heading_preservation_and_breadcrumbs():
    """Chunker must preserve multi-level heading hierarchy in section_path and prefix."""
    content = """# Drug Title

## Section A
Intro to section A.

### Subsection A.1
Detail for subsection A.1.

#### Sub-subsection A.1.a
Granular clinical detail.
"""
    doc = Document(
        file_path=Path("/tmp/test.md"),
        filename="test.md",
        title="Drug Title",
        content=content,
        source_type="DailyMed",
        drug_name="TestDrug"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=500)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 3
    paths = [c.section_path for c in chunks]
    assert "Drug Title > Section A" in paths
    assert "Drug Title > Section A > Subsection A.1" in paths
    assert "Drug Title > Section A > Subsection A.1 > Sub-subsection A.1.a" in paths

    # Verify contextual header prefix is prepended
    deep_chunk = [c for c in chunks if "Sub-subsection A.1.a" in c.section_path][0]
    assert deep_chunk.text.startswith("# Drug Title > Section A > Subsection A.1 > Sub-subsection A.1.a\n\n")


def test_chunker_empty_container_headers():
    """Chunker must not produce empty chunks when container headings have no direct text."""
    content = """# Drug Title

## Container Section With No Body
### Child Section
This child section has actual text.
"""
    doc = Document(
        file_path=Path("/tmp/test.md"),
        filename="test.md",
        title="Drug Title",
        content=content,
        source_type="DailyMed"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=500)
    chunks = chunker.chunk_document(doc)

    # Only 1 chunk should be produced (for Child Section), not 2
    assert len(chunks) == 1
    assert chunks[0].section_title == "Child Section"
    assert "Container Section With No Body > Child Section" in chunks[0].section_path
    assert chunks[0].text.strip() != ""


def test_chunker_short_section_preservation():
    """Chunker must preserve short, self-contained clinical sections intact without padding."""
    content = """# Drug Title

## 4 CONTRAINDICATIONS
Contraindicated in severe renal impairment.
"""
    doc = Document(
        file_path=Path("/tmp/test.md"),
        filename="test.md",
        title="Drug Title",
        content=content,
        source_type="DailyMed"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=600)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].token_count < 50
    assert "Contraindicated in severe renal impairment." in chunks[0].text


def test_chunker_table_preservation_and_header_repetition():
    """Chunker must repeat table header and separator rows when splitting an oversized table."""
    # Construct a large table that exceeds target_chunk_size
    header = "| Parameter | Normal Range | Adverse Level |\n|---|---|---|\n"
    rows = "\n".join(f"| Test {i} | Range {i} | Danger {i} |" for i in range(1, 40))
    table_content = f"# Clinical Lab\n\n## Lab Values\n{header}{rows}\n"

    doc = Document(
        file_path=Path("/tmp/table.md"),
        filename="table.md",
        title="Clinical Lab",
        content=table_content,
        source_type="DailyMed"
    )

    # Set a small target chunk size to force table partitioning
    chunker = MarkdownSectionChunker(target_chunk_size=120, overlap_tokens=0)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        # Every partition must contain the table header row and separator line
        assert "| Parameter | Normal Range | Adverse Level |" in chunk.text
        assert "|---|---|---|" in chunk.text


def test_chunker_line_ranges():
    """Chunker must accurately track 1-indexed start_line and end_line in source file."""
    content = """# Title

## Section 1
Line 4 content.
Line 5 content.

## Section 2
Line 8 content.
Line 9 content.
Line 10 content.
"""
    doc = Document(
        file_path=Path("/tmp/lines.md"),
        filename="lines.md",
        title="Title",
        content=content,
        source_type="DailyMed"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=500)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 2
    c1, c2 = chunks[0], chunks[1]

    assert c1.start_line == 4
    assert c1.end_line == 5

    assert c2.start_line == 8
    assert c2.end_line == 10


def test_chunker_overlap_behavior():
    """Chunker must carry forward trailing units when sub-chunking long sections."""
    # Generate paragraphs that require splitting
    p1 = "Paragraph 1: " + "alpha " * 50
    p2 = "Paragraph 2: " + "beta " * 50
    p3 = "Paragraph 3: " + "gamma " * 50
    content = f"# Doc\n\n## Long Section\n\n{p1}\n\n{p2}\n\n{p3}\n"

    doc = Document(
        file_path=Path("/tmp/overlap.md"),
        filename="overlap.md",
        title="Doc",
        content=content,
        source_type="DailyMed"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=120, overlap_tokens=40)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    # Second chunk should contain overlap from preceding content
    assert any("alpha" in c.text and "beta" in c.text for c in chunks) or \
           any("beta" in c.text and "gamma" in c.text for c in chunks)


def test_real_corpus_ingestion_integrity():
    """Integration test verifying full ingestion pipeline against actual app/data/corpus."""
    corpus_dir = Path("app/data/corpus")
    assert corpus_dir.exists()

    loader = CorpusLoader(corpus_dir)
    docs = loader.load_all()
    assert len(docs) == 21, "All 21 documents must load successfully"

    chunker = MarkdownSectionChunker(target_chunk_size=600, overlap_tokens=80)
    chunks = chunker.chunk_all(docs)

    assert len(chunks) > 400
    for chunk in chunks:
        assert chunk.chunk_id != ""
        assert chunk.source_filename.endswith(".md")
        assert chunk.section_title != ""
        assert chunk.section_path != ""
        assert chunk.text.strip() != ""
        assert 0 < chunk.token_count <= 600, f"Chunk {chunk.chunk_id} exceeds 600 tokens: {chunk.token_count}"
        assert chunk.start_line <= chunk.end_line
        assert 1 <= chunk.total_parts
        assert 0 <= chunk.part_index < chunk.total_parts


def test_chunker_code_block_headings_ignored():
    """Headings inside fenced code blocks must be treated as code body, not document sections."""
    content = """# Main Document

## Section 1
Introductory text.

```python
# This is a python comment, not a markdown heading
## This is also inside the code block
def sample():
    return 42
```

Trailing text in section 1.
"""
    doc = Document(
        file_path=Path("/tmp/code_test.md"),
        filename="code_test.md",
        title="Main Document",
        content=content,
        source_type="DailyMed"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=500)
    chunks = chunker.chunk_document(doc)

    # Only 1 chunk for Section 1 should be produced; python comments must not create new sections
    assert len(chunks) == 1
    assert chunks[0].section_title == "Section 1"
    assert "Section 1" in chunks[0].section_path
    assert "# This is a python comment, not a markdown heading" in chunks[0].text
    assert "def sample():" in chunks[0].text


def test_chunker_h5_h6_hierarchy():
    """Chunker must track full heading hierarchy up through H5 and H6."""
    content = """# Level 1
## Level 2
### Level 3
#### Level 4
##### Level 5
###### Level 6
Deeply nested clinical detail.
"""
    doc = Document(
        file_path=Path("/tmp/deep_test.md"),
        filename="deep_test.md",
        title="Level 1",
        content=content,
        source_type="DailyMed"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=500)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 1
    expected_path = "Level 1 > Level 2 > Level 3 > Level 4 > Level 5 > Level 6"
    assert chunks[0].section_path == expected_path
    assert chunks[0].section_title == "Level 6"
    assert chunks[0].text.startswith(f"# {expected_path}\n\n")


def test_chunker_strict_target_chunk_size():
    """Chunker must strictly enforce that no emitted chunk exceeds target_chunk_size."""
    # Create dense paragraphs of text
    content = "# Document\n\n## Section\n\n" + "\n\n".join(
        f"Paragraph {i}: " + ("detailed clinical monitoring data " * 30)
        for i in range(1, 15)
    )
    doc = Document(
        file_path=Path("/tmp/strict_test.md"),
        filename="strict_test.md",
        title="Document",
        content=content,
        source_type="DailyMed"
    )

    target_size = 250
    chunker = MarkdownSectionChunker(target_chunk_size=target_size, overlap_tokens=40)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= target_size, f"Chunk {c.chunk_id} ({c.token_count} toks) exceeded target {target_size}"


def test_split_table_total_parts_and_part_index():
    """Split tables must have total_parts matching partition count and 0-indexed part_index."""
    header = "| Drug | Dose | Response |\n|---|---|---|\n"
    rows = "\n".join(f"| Drug {i} | {i*10} mg | Clinical outcome {i} |" for i in range(1, 30))
    content = f"# Pharmacology\n\n## Dosage Table\n{header}{rows}\n"

    doc = Document(
        file_path=Path("/tmp/table_parts.md"),
        filename="table_parts.md",
        title="Pharmacology",
        content=content,
        source_type="DailyMed"
    )

    chunker = MarkdownSectionChunker(target_chunk_size=100, overlap_tokens=0)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    total_expected = len(chunks)
    for idx, c in enumerate(chunks):
        assert c.part_index == idx, f"Expected part_index {idx}, got {c.part_index}"
        assert c.total_parts == total_expected, f"Expected total_parts {total_expected}, got {c.total_parts}"
        assert c.token_count <= 100


def test_split_oversized_table_row():
    """A single table row exceeding the chunk budget must be partitioned into aligned continuation rows."""
    header = "| Drug | Detailed Mechanism and Adverse Warnings |\n|---|---|\n"
    # Row with ~400 tokens of dense clinical narrative
    huge_row = "| Warfarin | " + ("Massive clinical description of coagulation cascades, CYP interactions, and hemorrhage risks. " * 25) + "|"
    content = f"# Clinical Data\n\n## Interaction Matrix\n{header}{huge_row}\n"

    doc = Document(
        file_path=Path("/tmp/oversized_row.md"),
        filename="oversized_row.md",
        title="Clinical Data",
        content=content,
        source_type="DailyMed"
    )

    # Set tight target size of 150 tokens to force the single row to partition
    chunker = MarkdownSectionChunker(target_chunk_size=150, overlap_tokens=0)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.token_count <= 150, f"Chunk {c.chunk_id} exceeded target: {c.token_count}"
        assert c.part_index == idx
        assert c.total_parts == len(chunks)
        # Verify table structure is maintained with headers
        assert "| Drug | Detailed Mechanism and Adverse Warnings |" in c.text
        assert "|---|---|" in c.text

    # Second partition onwards must include continuation markers
    assert any("(cont." in c.text for c in chunks[1:])


def test_chunk_ids_unique_and_deterministic():
    """Chunk IDs must be strictly unique and deterministic across multiple runs."""
    loader = CorpusLoader("app/data/corpus")
    docs = loader.load_all()
    chunker = MarkdownSectionChunker(target_chunk_size=600, overlap_tokens=80)

    run1_chunks = chunker.chunk_all(docs)
    run2_chunks = chunker.chunk_all(docs)

    # Check uniqueness
    ids1 = [c.chunk_id for c in run1_chunks]
    assert len(ids1) == len(set(ids1)), "Duplicate chunk IDs found in run 1!"

    ids2 = [c.chunk_id for c in run2_chunks]
    assert len(ids2) == len(set(ids2)), "Duplicate chunk IDs found in run 2!"

    # Check determinism across runs
    assert ids1 == ids2, "Chunk IDs are not deterministic across runs!"
    for c1, c2 in zip(run1_chunks, run2_chunks):
        assert c1.chunk_id == c2.chunk_id
        assert c1.part_index == c2.part_index
        assert c1.total_parts == c2.total_parts
        assert c1.token_count == c2.token_count
        assert c1.text == c2.text
