# src/vector_store/neon_db_manager.py

from typing import List

from langchain_community.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Import the configuration variables we defined earlier
from src.config import GEMINI_API_KEY, NEON_DATABASE_URL


class NeonDBManager:
    """
    A manager class to handle all interactions with the Neon vector database
    using PGVector.
    """
    
    # The name of the table in the database where embeddings will be stored.
    # LangChain calls this a "collection".
    COLLECTION_NAME = "codebase_embeddings"

    def __init__(self):
        """
        Initializes the manager with the embedding model and database connection string.
        """
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not configured. Please check your .env file.")
        
        # Initialize the embedding model from Google
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
        
        self.connection_string = NEON_DATABASE_URL
        self.vector_store = None

    def create_vector_store_from_documents(self, documents: List[Document]):
        """
        Creates embeddings for the given documents and stores them in the Neon database.
        This method will create a new collection or overwrite an existing one.

        Args:
            documents (List[Document]): A list of chunked documents from the loader.
        """
        print("Starting the embedding and ingestion process...")
        
        try:
            # This is the core LangChain function that does the heavy lifting:
            # 1. It takes the documents.
            # 2. It uses the provided embedding function (self.embeddings) to convert text to vectors.
            # 3. It connects to the database using the connection string.
            # 4. It stores the embeddings and document metadata in the specified collection.
            self.vector_store = PGVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.COLLECTION_NAME,
                connection_string=self.connection_string,
            )
            print(f"Successfully created vector store and ingested {len(documents)} chunks into the '{self.COLLECTION_NAME}' collection.")
        except Exception as e:
            print(f"An error occurred during vector store creation: {e}")
            raise

    def get_retriever(self, search_k: int = 4):
        """
        Initializes a connection to an existing vector store and returns a retriever object.
        A retriever is a LangChain interface for fetching relevant documents.

        Args:
            search_k (int): The number of top relevant documents to retrieve.

        Returns:
            A LangChain retriever object.
        """
        if not self.vector_store:
            # If the vector store hasn't been created in this session,
            # this initializes a connection to an *existing* collection in the database.
            print(f"Connecting to existing vector store collection: '{self.COLLECTION_NAME}'")
            self.vector_store = PGVector(
                collection_name=self.COLLECTION_NAME,
                connection_string=self.connection_string,
                embedding_function=self.embeddings,
            )
        
        if not self.vector_store:
             raise ValueError("Vector store is not initialized. Ingest documents first.")

        # The as_retriever() method creates an object that can be used to
        # fetch documents based on a query.
        return self.vector_store.as_retriever(
            search_kwargs={'k': search_k}
        )