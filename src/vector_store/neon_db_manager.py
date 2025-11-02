# src/vector_store/neon_db_manager.py

from typing import List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    CHROMA_API_KEY,
    CHROMA_DATABASE,
    CHROMA_TENANT_ID,
)


class ChromaDBManager:
    """Manage interactions with a Chroma Cloud collection."""

    DEFAULT_COLLECTION_PREFIX = "codebase_"

    def __init__(self, collection_name: Optional[str] = None):
        """Initialize embeddings and the Chroma Cloud client.

        Args:
            collection_name: Name of the Chroma collection to target. A name
                will be generated if one is not supplied.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT_ID,
            database=CHROMA_DATABASE,
        )
        self.collection_name = collection_name or self._generate_collection_name()
        self.vector_store = None

    def _generate_collection_name(self) -> str:
        import uuid

        return f"{self.DEFAULT_COLLECTION_PREFIX}{uuid.uuid4().hex[:8]}"

    def _ensure_vector_store(self):
        if not self.vector_store:
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self.vector_store

    def create_vector_store_from_documents(self, documents: List[Document]):
        """Create or replace the cloud collection with fresh documents."""
        print("Starting the embedding and ingestion process...")

        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                client=self.client,
                collection_name=self.collection_name,
            )
            print(
                f"Successfully uploaded {len(documents)} chunks to the '{self.collection_name}' collection."
            )
        except Exception as exc:
            print(f"An error occurred during vector store creation: {exc}")
            raise

    def get_retriever(self, search_k: int = 4):
        """Return a retriever connected to the Chroma Cloud collection."""
        store = self._ensure_vector_store()
        return store.as_retriever(search_kwargs={"k": search_k})
