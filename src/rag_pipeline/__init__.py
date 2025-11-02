"""RAG pipeline package: high-level orchestration of retrieval + generation."""

from .qa_chain import create_rag_chain

__all__ = ["create_rag_chain"]
