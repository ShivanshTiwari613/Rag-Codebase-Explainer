# app/main_ui.py

import os
import tempfile
from pathlib import Path
import sys
import streamlit as st

# Ensure the project root is on sys.path so `src` can be imported when Streamlit
# runs the script from the `app/` folder.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import our modular backend components
from src.data_processing.loader import load_and_chunk_codebase
from src.rag_pipeline.qa_chain import create_rag_chain
from src.vector_store.neon_db_manager import NeonDBManager

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Codebase Q&A with Gemini & NeonDB",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Codebase Q&A with Gemini & NeonDB")
st.markdown("""
Welcome! This application allows you to chat with your own codebase.

**How it works:**
1.  Upload your codebase as a single `.zip` file or an individual code file.
2.  The application will process, chunk, and embed the code into a Neon vector database.
3.  Ask any question about your code, and the RAG pipeline will retrieve relevant context to provide an answer.
""")

# --- Session State Initialization ---
# This is crucial to maintain state across user interactions (e.g., asking multiple questions)
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "db_manager" not in st.session_state:
    st.session_state.db_manager = NeonDBManager()
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Upload Your Codebase")
    uploaded_file = st.file_uploader(
        "Upload a .zip file or a single code file",
        type=['zip', 'py', 'js', 'html', 'css', 'md', 'java', 'c', 'cpp']
    )

    if uploaded_file is not None and not st.session_state.processing_complete:
        st.info("Processing codebase... This may take a few moments.")
        
        # Save uploaded file to a temporary location to get a stable path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            with st.spinner("Step 1/3: Loading and chunking documents..."):
                chunked_docs = load_and_chunk_codebase(tmp_file_path)

            with st.spinner("Step 2/3: Embedding and ingesting into NeonDB... (This is the longest step)"):
                st.session_state.db_manager.create_vector_store_from_documents(chunked_docs)

            with st.spinner("Step 3/3: Creating the RAG chain..."):
                retriever = st.session_state.db_manager.get_retriever()
                st.session_state.rag_chain = create_rag_chain(retriever)
            
            st.session_state.processing_complete = True
            st.success("Codebase processed successfully! You can now ask questions.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        finally:
            # Clean up the temporary file
            os.remove(tmp_file_path)

# --- Main Chat Interface ---
st.header("2. Ask a Question")

if st.session_state.processing_complete:
    user_question = st.text_input("Enter your question about the codebase:")

    if user_question:
        with st.spinner("Retrieving context and generating answer..."):
            try:
                # Invoke the RAG chain with the user's question
                response = st.session_state.rag_chain.invoke(user_question)
                answer = response.get("answer", "No answer found.")
                source_documents = response.get("source_documents", [])

                st.markdown("### Answer")
                st.markdown(answer)

                # Display the source documents used for the answer
                if source_documents:
                    with st.expander("Show Sources"):
                        for doc in source_documents:
                            st.markdown(f"**Source: `{doc.metadata.get('source', 'N/A')}`**")
                            st.code(doc.page_content, language='text')

            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
else:
    st.warning("Please upload a codebase in the sidebar to begin.")