# app/main_ui.py

import os
import tempfile
from pathlib import Path
import sys
import uuid
import streamlit as st
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# Ensure the project root is on sys.path so src can be imported when Streamlit
# runs the script from the app/ folder.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import our modular backend components
from src.data_processing.loader import load_and_chunk_codebase
from src.rag_pipeline.qa_chain import create_rag_chain
from src.vector_store.chroma_cloud_manager import ChromaDBManager


# --- Helper functions ---
def _generate_default_collection_name() -> str:
    return f"codebase_{uuid.uuid4().hex[:8]}"


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Codebase Q&A with Gemini & Chroma",
    page_icon="?",
    layout="wide"
)

st.title("Codebase Q&A with Gemini & Chroma")
st.markdown("""
Welcome! This application allows you to chat with your own codebase.

**How it works:**
1.  Upload your codebase as a single .zip file or an individual code file.
2.  The application will process, chunk, and embed the code into your chosen Chroma Cloud collection.
3.  Ask any question about your code, and the RAG pipeline will retrieve relevant context to provide an answer.
""")

# --- Session State Initialization ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "db_manager" not in st.session_state:
    st.session_state.db_manager = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None
if "collection_input_seed" not in st.session_state:
    st.session_state.collection_input_seed = _generate_default_collection_name()
if "collection_input" not in st.session_state:
    st.session_state.collection_input = st.session_state.collection_input_seed
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "vector_weight" not in st.session_state:
    st.session_state.vector_weight = 0.7


def reset_for_new_upload():
    st.session_state.rag_chain = None
    st.session_state.db_manager = None
    st.session_state.processing_complete = False
    st.session_state.active_collection = None
    st.session_state.collection_input_seed = _generate_default_collection_name()
    st.session_state.pop("collection_input", None)
    st.session_state.chat_history = []
    st.session_state.bm25_retriever = None


# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload Your Codebase")
    st.text_input(
        "Collection name",
        key="collection_input",
        help="Chunks for this upload will be stored in this Chroma collection.",
    )
    vector_weight = st.slider(
        "Vector retriever weight",
        min_value=0.1,
        max_value=0.9,
        value=st.session_state.vector_weight,
        step=0.05,
        help="Higher values favor semantic search; lower values favor keyword search.",
    )
    st.session_state.vector_weight = vector_weight

    uploaded_file = st.file_uploader(
        "Upload a .zip file or a single code file",
        type=['zip', 'py', 'js', 'html', 'css', 'md', 'java', 'c', 'cpp']
    )

    if uploaded_file is not None and not st.session_state.processing_complete:
        st.info("Processing codebase... This may take a few moments.")

        collection_name = st.session_state.collection_input.strip() or _generate_default_collection_name()
        st.session_state.active_collection = collection_name
        st.session_state.db_manager = ChromaDBManager(collection_name=collection_name)

        # Save uploaded file to a temporary location to get a stable path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            with st.spinner("Step 1/3: Loading and chunking documents..."):
                chunked_docs = load_and_chunk_codebase(tmp_file_path)

            with st.spinner("Step 2/3: Embedding and uploading to Chroma Cloud... (This is the longest step)"):
                st.session_state.db_manager.create_vector_store_from_documents(chunked_docs)

            with st.spinner("Step 3/3: Creating the RAG chain..."):
                vector_retriever = st.session_state.db_manager.get_retriever()
                bm25_retriever = BM25Retriever.from_documents(chunked_docs)
                st.session_state.bm25_retriever = bm25_retriever
                weight = st.session_state.vector_weight
                ensemble = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[weight, 1 - weight],
                )
                st.session_state.rag_chain = create_rag_chain(ensemble)

            st.session_state.processing_complete = True
            st.success(
                f"Codebase processed successfully! Collection: {collection_name}. You can now ask questions."
            )

        except Exception as e:
            reset_for_new_upload()
            st.error(f"An error occurred during processing: {e}")
        finally:
            # Clean up the temporary file
            os.remove(tmp_file_path)

    if st.session_state.processing_complete:
        st.caption(f"Current collection: {st.session_state.active_collection}")
        if st.button("Start a new upload"):
            reset_for_new_upload()

# --- Main Chat Interface ---
st.header("2. Ask a Question")

if st.session_state.processing_complete and st.session_state.rag_chain:
    if st.session_state.chat_history:
        st.subheader("Conversation so far")
        for idx, (prev_q, prev_a) in enumerate(st.session_state.chat_history, start=1):
            st.markdown(f"**You {idx}:** {prev_q}")
            st.markdown(f"**Assistant {idx}:** {prev_a}")

    user_question = st.text_input("Enter your question about the codebase:")

    if user_question:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Invoke the RAG chain with the user's question
                response = st.session_state.rag_chain.invoke(
                    {
                        "question": user_question,
                        "chat_history": st.session_state.chat_history,
                    }
                )
                answer = response.get("answer", "No answer found.")
                source_documents = response.get("source_documents", [])

                st.markdown("### Answer")
                st.markdown(answer)

                st.session_state.chat_history.append((user_question, answer))
                st.session_state.chat_history = st.session_state.chat_history[-10:]

                # Display the source documents used for the answer
                if source_documents:
                    with st.expander("Show Sources"):
                        for doc in source_documents:
                            st.markdown(f"**Source: {doc.metadata.get('source', 'N/A')}**")
                            st.code(doc.page_content, language='text')

            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
else:
    st.warning("Please upload a codebase in the sidebar to begin.")


