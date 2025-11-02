# Codebase RAG Project (minimal scaffold)

This repository contains a minimal scaffold for a Retrieval-Augmented Generation (RAG)
project. It provides:

- `app/main_ui.py`: a small Streamlit UI to ingest a codebase and ask questions.
- `src/config.py`: environment-variable based configuration helper.
- `src/data_processing/loader.py`: file loader + simple character-based chunker.
- `src/vector_store/neon_db_manager.py`: a lightweight Neon manager scaffold (stub).
- `src/rag_pipeline/qa_chain.py`: a minimal RAG pipeline that ingests and answers.

How to run (basic):

1. Create a virtualenv and install requirements from `requirements.txt` (if you add Streamlit and other libs).
2. Run the UI:

   streamlit run app/main_ui.py

Notes:
- The Neon manager is currently a stub. Integrate your vector DB SDK in
  `src/vector_store/neon_db_manager.py`.
- The chunker is character-based; for production you should switch to a token-aware splitter.
