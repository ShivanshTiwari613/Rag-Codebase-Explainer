# src/data_processing/loader.py

import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

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


def load_and_chunk_codebase(file_path: str) -> List[Document]:
    """
    Loads a codebase from a file path (can be a single file or a zip archive),
    and splits the documents into chunks suitable for a RAG pipeline.

    Args:
        file_path (str): The path to the user-uploaded file.

    Returns:
        List[Document]: A list of chunked documents, ready for embedding.
    """
    documents = []
    
    if zipfile.is_zipfile(file_path):
        # If the file is a zip archive, extract it to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Use DirectoryLoader to load all supported files from the extracted content
            # We specify a glob to include a wide range of common code/text files
            loader = DirectoryLoader(
                temp_dir, 
                glob="**/*.*",
                show_progress=True,
                use_multithreading=True,
                loader_cls=TextLoader  # Use TextLoader for all files
            )
            documents = loader.load()

    else:
        # If it's a single file, load it directly
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

    # Get the file extension to determine the language for the splitter
    file_extension = Path(documents[0].metadata.get("source", "")).suffix
    language = LANGUAGE_MAP.get(file_extension, "python") # Default to python

    # Initialize a text splitter that is aware of code structures
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=2000, chunk_overlap=200
    )
    
    chunked_documents = text_splitter.split_documents(documents)
    
    print(f"Successfully loaded and chunked {len(documents)} document(s) into {len(chunked_documents)} chunks.")

    return chunked_documents
