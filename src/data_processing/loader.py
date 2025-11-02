# src/data_processing/loader.py

import ast
import re
import tempfile
import zipfile
from bisect import bisect_right
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import tiktoken
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define a mapping from file extensions to language names for the splitter
# This helps the splitter use appropriate separators for different code types.
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "js",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".sh": "bash",
    "Dockerfile": "docker",
}

TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
DEFAULT_CHUNK_SIZE_TOKENS = 1000
DEFAULT_CHUNK_OVERLAP_TOKENS = 120


def _token_length(text: str) -> int:
    return len(TOKEN_ENCODING.encode(text))


def _compute_line_starts(text: str) -> List[int]:
    starts = [0]
    offset = 0
    for line in text.splitlines(keepends=True):
        offset += len(line)
        starts.append(offset)
    return starts


def _line_for_index(char_index: int, line_starts: List[int]) -> int:
    if not line_starts:
        return 1
    position = bisect_right(line_starts, char_index) - 1
    return max(1, position + 1)


def _get_language_for_path(path: Optional[str]) -> str:
    if not path:
        return "text"
    suffix = Path(path).suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return LANGUAGE_MAP.get(suffix, "text")


def _relative_path(source: Optional[str], base_dir: Optional[Path]) -> str:
    if not source:
        return "unknown"
    try:
        source_path = Path(source).resolve()
        if base_dir:
            base_resolved = base_dir.resolve()
            return str(source_path.relative_to(base_resolved).as_posix())
    except Exception:
        pass
    return str(Path(source).resolve().as_posix())


def _python_symbol_path(lines: List[str], target_line: int) -> Optional[str]:
    source = "\n".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    class _Visitor(ast.NodeVisitor):
        def __init__(self, line: int) -> None:
            self.line = line
            self.stack: List[str] = []
            self.found: Optional[str] = None

        def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
            lineno = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", lineno)
            if lineno is not None and end is not None:
                if lineno <= self.line <= end:
                    super().generic_visit(node)
            else:
                super().generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
            if node.lineno <= self.line <= getattr(node, "end_lineno", node.lineno):
                self.stack.append(node.name)
                self.found = ".".join(self.stack)
                super().generic_visit(node)
                self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
            if node.lineno <= self.line <= getattr(node, "end_lineno", node.lineno):
                self.stack.append(node.name)
                super().generic_visit(node)
                self.stack.pop()

    visitor = _Visitor(target_line)
    visitor.visit(tree)
    return visitor.found


JS_FUNCTION_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_$]+)"),
    re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z0-9_$]+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"),
    re.compile(r"^\s*(?:export\s+)?([A-Za-z0-9_$]+)\s*=\s*function\b"),
]
JS_CLASS_PATTERNS = [
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z0-9_$]+)"),
]

JAVA_METHOD_PATTERNS = [
    re.compile(r"^\s*(?:public|protected|private|static|final|synchronized|abstract|native|transient)?\s+[A-Za-z0-9_<>\[\]]+\s+([A-Za-z0-9_]+)\s*\("),
]
JAVA_CLASS_PATTERNS = [
    re.compile(r"^\s*(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?class\s+([A-Za-z0-9_]+)"),
    re.compile(r"^\s*(?:public|protected|private)?\s*interface\s+([A-Za-z0-9_]+)"),
]


def _regex_symbol_path(
    lines: List[str],
    start_line: int,
    method_patterns: Iterable[re.Pattern[str]],
    container_patterns: Iterable[re.Pattern[str]],
) -> Optional[str]:
    method: Optional[str] = None
    container: Optional[str] = None
    for index in range(start_line - 1, -1, -1):
        text = lines[index].strip()
        if method is None:
            for pattern in method_patterns:
                match = pattern.match(text)
                if match:
                    method = match.group(1)
                    break
        if container is None:
            for pattern in container_patterns:
                match = pattern.match(text)
                if match:
                    container = match.group(1)
                    break
        if method and container:
            break
    parts = [p for p in (container, method) if p]
    return ".".join(parts) if parts else None


def _infer_symbol_path(language: str, lines: List[str], start_line: int) -> Optional[str]:
    if start_line <= 0:
        return None
    if language == "python":
        return _python_symbol_path(lines, start_line)
    if language in {"js", "typescript", "jsx", "tsx"}:
        return _regex_symbol_path(lines, start_line, JS_FUNCTION_PATTERNS, JS_CLASS_PATTERNS)
    if language in {"java", "csharp", "cpp", "c"}:
        return _regex_symbol_path(lines, start_line, JAVA_METHOD_PATTERNS, JAVA_CLASS_PATTERNS)
    return None


def _infer_markdown_heading(lines: List[str], start_line: int) -> Tuple[Optional[str], Optional[int]]:
    for index in range(start_line - 1, -1, -1):
        text = lines[index].rstrip()
        if text.startswith("#"):
            level = len(text) - len(text.lstrip("#"))
            title = text[level:].strip()
            return (title or None, level)
    return (None, None)


def _build_splitter(language: str) -> RecursiveCharacterTextSplitter:
    try:
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=DEFAULT_CHUNK_SIZE_TOKENS,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
            length_function=_token_length,
            add_start_index=True,
        )
    except ValueError:
        return RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE_TOKENS,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP_TOKENS,
            length_function=_token_length,
            add_start_index=True,
        )

def _load_with_fallback(path: str) -> List[Document]:
    """Load a single file using TextLoader with encoding fallbacks."""
    loader = TextLoader(
        path,
        encoding="utf-8",
        autodetect_encoding=True,
    )
    return loader.load()


def load_and_chunk_codebase(file_path: str) -> List[Document]:
    """
    Loads a codebase from a file path (can be a single file or a zip archive),
    and splits the documents into chunks suitable for a RAG pipeline.

    Args:
        file_path (str): The path to the user-uploaded file.

    Returns:
        List[Document]: A list of chunked documents, ready for embedding.
    """
    documents: List[Document] = []
    base_dir: Optional[Path] = None

    if zipfile.is_zipfile(file_path):
        # If the file is a zip archive, extract it to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            # Use DirectoryLoader to load all supported files from the extracted content
            loader = DirectoryLoader(
                temp_dir,
                glob="**/*.*",
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader,
                loader_kwargs={
                    "encoding": "utf-8",
                    "autodetect_encoding": True,
                },
                silent_errors=True,
            )
            documents = loader.load()
            base_dir = Path(temp_dir)

    else:
        # If it's a single file, load it directly
        documents = _load_with_fallback(file_path)
        base_dir = Path(file_path).resolve().parent

    if not documents:
        raise ValueError("No readable documents were loaded from the provided file.")

    chunked_documents: List[Document] = []

    for doc in documents:
        source_path = doc.metadata.get("source")
        language = _get_language_for_path(source_path)
        splitter = _build_splitter(language)

        base_metadata = dict(doc.metadata)
        relative = _relative_path(source_path, base_dir)

        line_starts = _compute_line_starts(doc.page_content)
        lines = doc.page_content.splitlines()

        chunks = splitter.create_documents([doc.page_content], metadatas=[base_metadata])
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            start_index = metadata.pop("start_index", None)
            if start_index is not None:
                start_line = _line_for_index(start_index, line_starts)
                end_index = start_index + len(chunk.page_content)
                end_line = _line_for_index(max(end_index - 1, start_index), line_starts)
            else:
                start_line = end_line = None

            symbol_path = _infer_symbol_path(language, lines, start_line or 0)
            md_title: Optional[str] = None
            md_level: Optional[int] = None
            if language == "markdown":
                md_title, md_level = _infer_markdown_heading(lines, start_line or 0)
                if not symbol_path and md_title:
                    symbol_path = md_title

            metadata.update(
                {
                    "path": relative,
                    "language": language,
                    "chunk_idx": idx,
                    "start_line": start_line,
                    "end_line": end_line,
                    "symbol_path": symbol_path,
                }
            )

            if language == "markdown":
                metadata["section_title"] = md_title
                metadata["section_level"] = md_level

            chunk.metadata = metadata
            chunked_documents.append(chunk)

    print(f"Successfully loaded and chunked {len(documents)} document(s) into {len(chunked_documents)} chunks.")

    return chunked_documents

