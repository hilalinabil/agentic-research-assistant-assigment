"""
Hierarchical, section-aware Markdown chunker for pharmacological documents.
Preserves heading hierarchy (H1-H6), code blocks, line ranges, table column headers, and clinical precision.
"""
from pathlib import Path
import re
import logging
from typing import List, Optional, Tuple, Dict, Any

from app.schemas.document import Document, DocumentChunk

logger = logging.getLogger(__name__)

# Initialize tokenizer with graceful fallback
try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    IS_EXACT_TOKENIZER = True
except Exception as e:
    logger.warning(f"tiktoken initialization failed ({e}); falling back to regex token estimator.")
    _TIKTOKEN_ENCODER = None
    IS_EXACT_TOKENIZER = False


def count_tokens(text: str) -> int:
    """
    Returns exact BPE token count using tiktoken (cl100k_base),
    or falls back to an approximate regex token count if tiktoken is unavailable.
    """
    if _TIKTOKEN_ENCODER is not None:
        return len(_TIKTOKEN_ENCODER.encode(text))
    # Approximation fallback: counts words and individual punctuation marks
    return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


class MarkdownSectionChunker:
    """
    Splits pharmacological documents into discrete, grounded chunks:
    - Preserves heading breadcrumb hierarchy (# H1 > H2 > H3 > H4 > H5 > H6)
    - Ignores Markdown headings inside fenced code blocks (``` or ~~~)
    - Strictly enforces target_chunk_size ceiling
    - Preserves Markdown tables and repeats column headers if a table must be split
    - Tracks source line numbers (start_line, end_line)
    - Prevents empty chunks from container headers
    """

    def __init__(
        self,
        target_chunk_size: int = 600,
        overlap_tokens: int = 80,
    ):
        self.target_chunk_size = target_chunk_size
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Processes a single Document into a list of DocumentChunk instances.
        """
        raw_lines = document.content.splitlines()
        if not raw_lines:
            return []

        # 1. Parse document into hierarchical sections with line numbers
        sections = self._parse_sections(raw_lines)
        if not sections:
            # Fallback if document has no Markdown headers
            sections = [{
                "heading": document.title,
                "section_path": document.title,
                "body_lines": [(l, idx) for idx, l in enumerate(raw_lines, 1)],
                "start_line": 1,
                "end_line": len(raw_lines),
            }]

        chunks: List[DocumentChunk] = []
        file_stem = Path(document.filename).stem

        chunk_counter = 0

        for sec_idx, sec in enumerate(sections):
            body_lines = sec["body_lines"]
            if not body_lines:
                # Skip container headers that have no direct body text
                continue

            section_path = sec["section_path"]
            header_prefix = f"# {section_path}\n\n"
            prefix_tokens = count_tokens(header_prefix)

            # Reconstruct section body text
            body_text = "\n".join(line for line, _ in body_lines).strip()
            if not body_text:
                continue

            # Strip leading/trailing blank lines for precise content line boundary
            non_blank_lines = [item for item in body_lines if item[0].strip()]
            sec_start_line = non_blank_lines[0][1] if non_blank_lines else body_lines[0][1]
            sec_end_line = non_blank_lines[-1][1] if non_blank_lines else body_lines[-1][1]

            full_chunk_text = header_prefix + body_text
            exact_total_tokens = count_tokens(full_chunk_text)

            chunk_metadata = {
                "source_type": document.source_type,
                "exact_tokenizer": IS_EXACT_TOKENIZER,
            }
            if "encoding_warning" in document.metadata:
                chunk_metadata["encoding_warning"] = document.metadata["encoding_warning"]

            # Check if entire section fits in a single chunk strictly under target_chunk_size
            if exact_total_tokens <= self.target_chunk_size:
                chunk_counter += 1
                chunks.append(DocumentChunk(
                    chunk_id=f"{file_stem}_chunk{chunk_counter:03d}",
                    source_filename=document.filename,
                    drug_name=document.drug_name,
                    section_title=sec["heading"],
                    section_path=section_path,
                    text=full_chunk_text,
                    token_count=exact_total_tokens,
                    part_index=0,
                    total_parts=1,
                    start_line=sec_start_line,
                    end_line=sec_end_line,
                    metadata=chunk_metadata
                ))
            else:
                # Sub-chunk long section while preserving units and tables
                sub_chunks = self._subchunk_section(
                    file_stem=file_stem,
                    document=document,
                    sec=sec,
                    body_lines=body_lines,
                    header_prefix=header_prefix,
                    prefix_tokens=prefix_tokens,
                    start_counter=chunk_counter + 1,
                    base_metadata=chunk_metadata
                )
                chunk_counter += len(sub_chunks)
                chunks.extend(sub_chunks)

        return chunks

    def chunk_all(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Chunks a list of documents.
        """
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Generated {len(all_chunks)} chunks across {len(documents)} documents")
        return all_chunks

    def _parse_sections(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parses Markdown lines into sections, tracking active heading hierarchy (H1-H6)
        and line numbers while properly ignoring headings inside fenced code blocks.
        """
        sections: List[Dict[str, Any]] = []
        heading_stack: Dict[int, str] = {}
        current_header = ""
        current_body_lines: List[Tuple[str, int]] = []
        current_start_line = 1
        in_code_block = False

        def flush_section():
            if current_body_lines or current_header:
                path_parts = [heading_stack[k] for k in sorted(heading_stack.keys())]
                section_path = " > ".join(path_parts) if path_parts else current_header
                
                content_lines = [item for item in current_body_lines if item[0].strip()]
                start_l = content_lines[0][1] if content_lines else current_start_line
                end_l = content_lines[-1][1] if content_lines else current_start_line

                sections.append({
                    "heading": current_header,
                    "section_path": section_path,
                    "body_lines": current_body_lines.copy(),
                    "start_line": start_l,
                    "end_line": end_l,
                })

        for idx, line in enumerate(lines, 1):
            stripped = line.strip()

            # Toggle fenced code blocks (``` or ~~~)
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_block = not in_code_block
                current_body_lines.append((line, idx))
                continue

            # Check for Markdown heading only when outside code blocks
            if not in_code_block:
                m = re.match(r"^(#{1,6})\s+(.*)$", line)
                if m:
                    lvl = len(m.group(1))
                    title = m.group(2).strip()

                    flush_section()
                    current_body_lines.clear()
                    current_header = title
                    current_start_line = idx

                    # Clear all levels >= current level and record current
                    heading_stack = {k: v for k, v in heading_stack.items() if k < lvl}
                    heading_stack[lvl] = title
                    continue

            current_body_lines.append((line, idx))

        flush_section()
        return sections

    def _subchunk_section(
        self,
        file_stem: str,
        document: Document,
        sec: Dict[str, Any],
        body_lines: List[Tuple[str, int]],
        header_prefix: str,
        prefix_tokens: int,
        start_counter: int,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Sub-chunks an oversized section into units (paragraphs, lists, or tables),
        repeating table headers if a table must be partitioned and strictly enforcing
        the target_chunk_size ceiling.
        """
        raw_units = self._extract_units(body_lines)
        if not raw_units:
            return []

        # Content budget strictly accounts for header prefix
        max_content_tokens = max(50, self.target_chunk_size - prefix_tokens)

        # Flatten units so that any unit exceeding max_content_tokens is broken down
        units: List[Dict[str, Any]] = []
        for u in raw_units:
            if u["token_count"] <= max_content_tokens or u.get("is_table"):
                units.append(u)
            else:
                units.extend(self._split_oversized_unit(u, max_content_tokens))

        chunks: List[DocumentChunk] = []
        current_units: List[Dict[str, Any]] = []
        part_idx = 0

        def create_chunk(accumulated_units: List[Dict[str, Any]], part_num: int) -> DocumentChunk:
            chunk_body = "\n\n".join(u["text"] for u in accumulated_units)
            start_l = accumulated_units[0]["start_line"]
            end_l = accumulated_units[-1]["end_line"]
            full_text = header_prefix + chunk_body
            total_toks = count_tokens(full_text)

            meta = base_metadata.copy()

            return DocumentChunk(
                chunk_id=f"{file_stem}_chunk{start_counter + part_num:03d}",
                source_filename=document.filename,
                drug_name=document.drug_name,
                section_title=sec["heading"],
                section_path=sec["section_path"],
                text=full_text,
                token_count=total_toks,
                part_index=part_num,
                total_parts=1,  # updated after loop
                start_line=start_l,
                end_line=end_l,
                metadata=meta
            )

        for unit in units:
            # If unit is an oversized table, split rows or single oversized row
            if unit.get("is_table") and count_tokens(header_prefix + unit["text"]) > self.target_chunk_size and len(unit.get("rows", [])) >= 1:
                # Flush previous units first
                if current_units:
                    chunks.append(create_chunk(current_units, part_idx))
                    part_idx += 1
                    current_units = []

                table_chunks = self._split_table_unit(
                    unit=unit,
                    max_tokens=max_content_tokens,
                    header_prefix=header_prefix,
                    file_stem=file_stem,
                    document=document,
                    sec=sec,
                    start_counter=start_counter + part_idx,
                    base_metadata=base_metadata
                )
                chunks.extend(table_chunks)
                part_idx += len(table_chunks)
                continue

            # Candidate chunk check: evaluate actual joined text token count
            candidate_units = current_units + [unit]
            candidate_body = "\n\n".join(u["text"] for u in candidate_units)
            candidate_tokens = count_tokens(header_prefix + candidate_body)

            if candidate_tokens <= self.target_chunk_size:
                current_units.append(unit)
            else:
                if current_units:
                    chunks.append(create_chunk(current_units, part_idx))
                    part_idx += 1

                    # Compute overlap from trailing units ensuring the next candidate fits
                    overlap_units: List[Dict[str, Any]] = []
                    for prev_u in reversed(current_units):
                        if prev_u.get("is_table"):
                            continue
                        candidate_overlap = [prev_u] + overlap_units
                        candidate_next_body = "\n\n".join(u["text"] for u in candidate_overlap + [unit])
                        candidate_overlap_tokens = count_tokens("\n\n".join(u["text"] for u in candidate_overlap))

                        if (candidate_overlap_tokens <= self.overlap_tokens and 
                            count_tokens(header_prefix + candidate_next_body) <= self.target_chunk_size):
                            overlap_units = candidate_overlap
                        else:
                            break

                    current_units = overlap_units + [unit]
                else:
                    # Single unit exceeds max budget on its own
                    current_units = [unit]
                    chunks.append(create_chunk(current_units, part_idx))
                    part_idx += 1
                    current_units = []

        if current_units:
            chunks.append(create_chunk(current_units, part_idx))
            part_idx += 1

        total_p = len(chunks)
        for idx, chk in enumerate(chunks):
            chk.part_index = idx
            chk.total_parts = total_p

        return chunks

    def _split_oversized_unit(self, unit: Dict[str, Any], max_tokens: int) -> List[Dict[str, Any]]:
        """
        Recursively splits an oversized unit into smaller units:
        1. By lines (for lists, single-newline blocks)
        2. By sentences (for long narrative paragraphs)
        3. By word groups if an individual sentence exceeds max_tokens
        """
        text = unit["text"]
        start_line = unit["start_line"]
        raw_lines = text.splitlines()

        if len(raw_lines) > 1:
            sub_units = []
            for offset, l in enumerate(raw_lines):
                l_clean = l.strip()
                if l_clean:
                    line_no = start_line + offset
                    sub_units.append({
                        "text": l_clean,
                        "start_line": line_no,
                        "end_line": line_no,
                        "token_count": count_tokens(l_clean),
                        "is_table": False,
                    })
            refined = []
            for su in sub_units:
                if su["token_count"] <= max_tokens:
                    refined.append(su)
                else:
                    refined.extend(self._split_oversized_unit(su, max_tokens))
            return refined

        # Single line: split by sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            sub_units = []
            for s in sentences:
                s_clean = s.strip()
                if s_clean:
                    sub_units.append({
                        "text": s_clean,
                        "start_line": start_line,
                        "end_line": unit["end_line"],
                        "token_count": count_tokens(s_clean),
                        "is_table": False,
                    })
            refined = []
            for su in sub_units:
                if su["token_count"] <= max_tokens:
                    refined.append(su)
                else:
                    refined.extend(self._split_oversized_unit(su, max_tokens))
            return refined

        # Single sentence exceeds max_tokens: chunk by words
        words = text.split()
        sub_units = []
        chunk_words = []
        for w in words:
            candidate = " ".join(chunk_words + [w])
            if count_tokens(candidate) <= max_tokens:
                chunk_words.append(w)
            else:
                if chunk_words:
                    sub_units.append({
                        "text": " ".join(chunk_words),
                        "start_line": start_line,
                        "end_line": unit["end_line"],
                        "token_count": count_tokens(" ".join(chunk_words)),
                        "is_table": False,
                    })
                chunk_words = [w]
        if chunk_words:
            sub_units.append({
                "text": " ".join(chunk_words),
                "start_line": start_line,
                "end_line": unit["end_line"],
                "token_count": count_tokens(" ".join(chunk_words)),
                "is_table": False,
            })
        return sub_units

    def _extract_units(self, body_lines: List[Tuple[str, int]]) -> List[Dict[str, Any]]:
        """
        Segments body lines into cohesive semantic units:
        - Markdown tables (grouped intact)
        - Paragraphs (blank-line separated)
        """
        units: List[Dict[str, Any]] = []
        idx = 0
        n = len(body_lines)

        while idx < n:
            line_text, line_no = body_lines[idx]

            # Check for Markdown table start
            if self._is_table_start(body_lines, idx):
                table_lines: List[Tuple[str, int]] = []
                while idx < n and "|" in body_lines[idx][0]:
                    table_lines.append(body_lines[idx])
                    idx += 1

                header_prefix = table_lines[0][0] + "\n" + table_lines[1][0]
                rows = table_lines[2:]
                full_table_text = "\n".join(l for l, _ in table_lines)

                units.append({
                    "text": full_table_text,
                    "start_line": table_lines[0][1],
                    "end_line": table_lines[-1][1],
                    "token_count": count_tokens(full_table_text),
                    "is_table": True,
                    "table_header_prefix": header_prefix,
                    "rows": rows,
                })
                continue

            # Skip blank lines between paragraphs
            if not line_text.strip():
                idx += 1
                continue

            # Accumulate paragraph
            para_lines: List[Tuple[str, int]] = []
            while idx < n and body_lines[idx][0].strip() and not self._is_table_start(body_lines, idx):
                para_lines.append(body_lines[idx])
                idx += 1

            if para_lines:
                para_text = "\n".join(l for l, _ in para_lines).strip()
                units.append({
                    "text": para_text,
                    "start_line": para_lines[0][1],
                    "end_line": para_lines[-1][1],
                    "token_count": count_tokens(para_text),
                    "is_table": False,
                })

        return units

    @staticmethod
    def _is_table_start(lines: List[Tuple[str, int]], idx: int) -> bool:
        """
        Detects if line at idx and line at idx+1 represent a Markdown table header and separator.
        """
        if idx + 1 < len(lines):
            l1 = lines[idx][0].strip()
            l2 = lines[idx + 1][0].strip()
            if "|" in l1 and "|" in l2 and ("---" in l2 or "-|-" in l2 or re.match(r"^\|?(\s*:?-+:?\s*\|)+\s*$", l2)):
                return True
        return False

    def _split_table_unit(
        self,
        unit: Dict[str, Any],
        max_tokens: int,
        header_prefix: str,
        file_stem: str,
        document: Document,
        sec: Dict[str, Any],
        start_counter: int,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Splits a table unit that exceeds target_chunk_size across row boundaries,
        repeating the table header row and separator in every partition.
        Handles oversized single rows by partitioning cells into aligned continuation rows,
        strictly guaranteeing that no chunk exceeds target_chunk_size.
        """
        chunks: List[DocumentChunk] = []
        table_hdr = unit["table_header_prefix"]

        raw_rows: List[Tuple[str, int]] = unit["rows"]
        # Pre-process rows: break any single oversized row into continuation rows
        rows: List[Tuple[str, int]] = []
        for r_text, r_line in raw_rows:
            single_row_candidate = header_prefix + table_hdr + "\n" + r_text
            if count_tokens(single_row_candidate) <= self.target_chunk_size:
                rows.append((r_text, r_line))
            else:
                logger.warning(
                    f"Oversized table row detected at line {r_line} in {document.filename}. "
                    "Partitioning cell content into continuation rows."
                )
                split_rows = self._split_oversized_table_row(
                    r_text=r_text,
                    r_line=r_line,
                    table_hdr=table_hdr,
                    header_prefix=header_prefix
                )
                rows.extend(split_rows)

        current_rows: List[Tuple[str, int]] = []
        part_idx = 0

        for r_text, r_line in rows:
            candidate_table = table_hdr + "\n" + "\n".join(t for t, _ in current_rows + [(r_text, r_line)])
            candidate_tokens = count_tokens(header_prefix + candidate_table)

            if candidate_tokens <= self.target_chunk_size:
                current_rows.append((r_text, r_line))
            else:
                if current_rows:
                    table_body = table_hdr + "\n" + "\n".join(t for t, _ in current_rows)
                    full_text = header_prefix + table_body
                    meta = base_metadata.copy()
                    meta["is_table_split"] = True

                    chunks.append(DocumentChunk(
                        chunk_id=f"{file_stem}_chunk{start_counter + part_idx:03d}",
                        source_filename=document.filename,
                        drug_name=document.drug_name,
                        section_title=sec["heading"],
                        section_path=sec["section_path"],
                        text=full_text,
                        token_count=count_tokens(full_text),
                        part_index=part_idx,
                        total_parts=1,  # updated below
                        start_line=unit["start_line"],
                        end_line=current_rows[-1][1],
                        metadata=meta
                    ))
                    part_idx += 1
                current_rows = [(r_text, r_line)]

        if current_rows:
            table_body = table_hdr + "\n" + "\n".join(t for t, _ in current_rows)
            full_text = header_prefix + table_body
            meta = base_metadata.copy()
            meta["is_table_split"] = True

            chunks.append(DocumentChunk(
                chunk_id=f"{file_stem}_chunk{start_counter + part_idx:03d}",
                source_filename=document.filename,
                drug_name=document.drug_name,
                section_title=sec["heading"],
                section_path=sec["section_path"],
                text=full_text,
                token_count=count_tokens(full_text),
                part_index=part_idx,
                total_parts=1,  # updated below
                start_line=unit["start_line"],
                end_line=current_rows[-1][1],
                metadata=meta
            ))

        # Strictly update total_parts and part_index for all partitions of this table
        total_p = len(chunks)
        for idx, chk in enumerate(chunks):
            chk.part_index = idx
            chk.total_parts = total_p

        return chunks

    def _split_oversized_table_row(
        self,
        r_text: str,
        r_line: int,
        table_hdr: str,
        header_prefix: str
    ) -> List[Tuple[str, int]]:
        """
        Splits a single table row that exceeds target_chunk_size when paired with table_hdr.
        Partitions the narrative cell(s) into aligned continuation rows tagged with (cont. X).
        """
        base_prefix = header_prefix + table_hdr + "\n"
        raw_cells = [c.strip() for c in r_text.strip().strip("|").split("|")]
        if not raw_cells:
            return [(r_text, r_line)]

        col0 = raw_cells[0]
        other_cells = raw_cells[1:] if len(raw_cells) > 1 else [""]
        long_text = " ".join(other_cells)
        words = long_text.split()

        sub_rows: List[Tuple[str, int]] = []
        chunk_words: List[str] = []
        part = 1

        for w in words:
            candidate_words = chunk_words + [w]
            prefix_label = f"{col0} (cont. {part})" if part > 1 else col0
            candidate_row = f"| {prefix_label} | {' '.join(candidate_words)} |"
            if count_tokens(base_prefix + candidate_row) <= self.target_chunk_size:
                chunk_words.append(w)
            else:
                if chunk_words:
                    prefix_label = f"{col0} (cont. {part})" if part > 1 else col0
                    row_str = f"| {prefix_label} | {' '.join(chunk_words)} |"
                    sub_rows.append((row_str, r_line))
                    part += 1
                chunk_words = [w]

        if chunk_words:
            prefix_label = f"{col0} (cont. {part})" if part > 1 else col0
            row_str = f"| {prefix_label} | {' '.join(chunk_words)} |"
            sub_rows.append((row_str, r_line))

        return sub_rows if sub_rows else [(r_text, r_line)]
