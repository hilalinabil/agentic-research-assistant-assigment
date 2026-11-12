"""
Document loader for the local pharmacology corpus.
Handles secure file discovery, safe reading, encoding management, and metadata extraction.
"""
from pathlib import Path
import re
import logging
from typing import List, Optional

from app.schemas.document import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".txt"}


class CorpusLoader:
    """
    Safely loads documents from a designated local corpus directory.
    Guarantees path confinement and explicit encoding management.
    """

    def __init__(self, corpus_dir: Path | str):
        # Resolve the provided path; if it does not exist relative to the current working directory,
        # attempt to resolve it relative to the project root (pharma-research-agent directory).
        resolved = Path(corpus_dir).expanduser().resolve()
        if not resolved.exists():
            # Assume the path is relative to the repository root (pharma-research-agent)
            project_root = Path(__file__).parents[2]  # pharma-research-agent directory
            resolved = (project_root / corpus_dir).resolve()
        self.corpus_dir = resolved
        if not self.corpus_dir.exists() or not self.corpus_dir.is_dir():
            raise FileNotFoundError(f"Corpus directory not found or not a directory: {self.corpus_dir}")
        self.encoding_warnings: List[str] = []

    def discover_files(self) -> List[Path]:
        """
        Discovers all supported document files (.md, .txt) within the corpus directory.
        Enforces strict path confinement to prevent directory traversal.
        """
        discovered = []
        for path in sorted(self.corpus_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                # Security: verify path is strictly confined within corpus_dir
                resolved_path = path.resolve()
                try:
                    resolved_path.relative_to(self.corpus_dir)
                except ValueError:
                    logger.warning(f"Skipping file outside corpus directory boundary: {path}")
                    continue
                discovered.append(path)
        return discovered

    def load_document(self, file_path: Path) -> Optional[Document]:
        """
        Loads a single document from disk safely, handling encodings and metadata.
        Returns None if the file cannot be read or is invalid.
        """
        resolved_path = file_path.resolve()
        try:
            resolved_path.relative_to(self.corpus_dir)
        except ValueError:
            logger.error(f"Security error: Attempted to load file outside corpus directory: {file_path}")
            return None

        # Read file with explicit encoding handling
        try:
            raw_bytes = resolved_path.read_bytes()
        except OSError as e:
            logger.error(f"I/O error reading file {resolved_path}: {e}")
            return None

        if len(raw_bytes) == 0:
            logger.warning(f"Skipping empty document: {resolved_path.name}")
            return None

        # Decode UTF-8 with explicit, recorded fallback
        encoding_used = "utf-8"
        encoding_warning: Optional[str] = None
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as err:
            warning_msg = (
                f"Encoding notice: UTF-8 decode failed for '{resolved_path.name}' at byte {err.start} "
                f"({err.reason}). Recovering via latin-1 fallback."
            )
            logger.warning(warning_msg)
            self.encoding_warnings.append(warning_msg)
            encoding_used = "latin-1 (fallback)"
            encoding_warning = warning_msg
            try:
                content = raw_bytes.decode("latin-1")
            except Exception as e:
                logger.error(f"Failed to decode {resolved_path.name} with fallback: {e}")
                return None

        filename = resolved_path.name
        source_type, drug_name = self._classify_document(filename)
        title = self._extract_title(content, filename)

        metadata = {
            "size_bytes": len(raw_bytes),
            "line_count": len(content.splitlines()),
            "file_extension": resolved_path.suffix.lower(),
            "encoding": encoding_used,
        }
        if encoding_warning:
            metadata["encoding_warning"] = encoding_warning

        return Document(
            file_path=resolved_path,
            filename=filename,
            title=title,
            content=content,
            source_type=source_type,
            drug_name=drug_name,
            metadata=metadata
        )

    def load_all(self) -> List[Document]:
        """
        Discovers and loads all valid documents in the corpus directory.
        """
        files = self.discover_files()
        documents = []
        for f in files:
            doc = self.load_document(f)
            if doc is not None:
                documents.append(doc)
        logger.info(f"Loaded {len(documents)} documents from {self.corpus_dir}")
        return documents

    @staticmethod
    def _classify_document(filename: str) -> tuple[str, Optional[str]]:
        """
        Extracts source_type and drug_name based on filename naming conventions.
        """
        lower = filename.lower()
        if "glossary" in lower:
            return "Glossary", None

        stem = Path(filename).stem
        if stem.startswith("dailymed-"):
            drug = stem[len("dailymed-"):].replace("-", " ").capitalize()
            return "DailyMed", drug
        elif stem.startswith("medlineplus-"):
            drug = stem[len("medlineplus-"):].replace("-", " ").capitalize()
            return "MedlinePlus", drug

        return "Unknown", None

    @staticmethod
    def _extract_title(content: str, default_filename: str) -> str:
        """
        Extracts the document title from the first H1 header outside code blocks,
        or defaults to the filename.
        """
        in_code = False
        for line in content.splitlines():
            s = line.strip()
            if s.startswith("```") or s.startswith("~~~"):
                in_code = not in_code
                continue
            if not in_code:
                m = re.match(r"^#\s+(.+)$", line)
                if m:
                    return m.group(1).strip()
        return default_filename
