"""
Data schemas for loaded documents and structured chunks.
"""
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents an authoritative document loaded from the local corpus.
    """
    file_path: Path = Field(..., description="Absolute or relative path to the source document")
    filename: str = Field(..., description="Basename of the file (e.g. 'dailymed-warfarin.md')")
    title: str = Field(..., description="Title of the document (typically from first H1 or filename)")
    content: str = Field(..., description="Raw text content of the document")
    source_type: str = Field(..., description="Source classification: 'DailyMed', 'MedlinePlus', or 'Glossary'")
    drug_name: Optional[str] = Field(None, description="Associated drug name if applicable")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata (e.g. URL, size)")


class DocumentChunk(BaseModel):
    """
    Represents a discrete, contextually grounded chunk of text with traceable metadata.
    """
    chunk_id: str = Field(..., description="Deterministic, unique identifier for the chunk")
    source_filename: str = Field(..., description="Original filename the chunk was extracted from")
    drug_name: Optional[str] = Field(None, description="Associated drug name")
    section_title: str = Field(..., description="Immediate heading title (e.g. '4 CONTRAINDICATIONS')")
    section_path: str = Field(
        ...,
        description="Full hierarchical heading breadcrumb (e.g. 'Warfarin (DailyMed label) > 4 CONTRAINDICATIONS')"
    )
    text: str = Field(..., description="Context-grounded text content of the chunk")
    token_count: int = Field(..., description="Exact BPE token count or approximation")
    part_index: int = Field(default=0, description="0-indexed partition sequence for split sections")
    total_parts: int = Field(default=1, description="Total number of partitions for this section")
    start_line: int = Field(..., description="1-indexed starting line number in the source file")
    end_line: int = Field(..., description="1-indexed ending line number in the source file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional chunk metadata")
