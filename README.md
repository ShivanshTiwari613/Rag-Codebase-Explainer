# Codebase RAG Project (minimal scaffold)

This repository contains a minimal scaffold for a Retrieval-Augmented Generation (RAG)
project. It provides:

- `app/main_ui.py`: a small Streamlit UI to ingest a codebase and ask questions.
- `src/config.py`: environment-variable based configuration helper.
- `src/data_processing/loader.py`: file loader + simple character-based chunker.
- `src/vector_store/neon_db_manager.py`: a Chroma Cloud manager implementation.
- `src/rag_pipeline/qa_chain.py`: a minimal RAG pipeline that ingests and answers.

How to run (basic):

1. Create a virtualenv and install requirements from `requirements.txt`.
2. Provide the following environment variables in your `.env` file:
   - `GEMINI_API_KEY`
   - `CHROMA_API_KEY`
   - `CHROMA_TENANT_ID`
   - `CHROMA_DATABASE`
3. Run the UI:

   streamlit run app/main_ui.py

Notes:
- Each ingestion creates (or reuses) a Chroma collection of your choice. Give it a memorable name in the sidebar before uploading.
- Embeddings are uploaded directly to your configured Chroma Cloud database; there is no local persistence.
- The chunker is character-based; for production you should switch to a token-aware splitter.

### Inspecting the Chroma store

Use the helper script to inspect stored chunks or run ad-hoc searches:

```bash
python scripts/inspect_chroma.py --list-collections
python scripts/inspect_chroma.py --collection codebase_abcd1234 --limit 5 --client cloud
python scripts/inspect_chroma.py --collection codebase_abcd1234 --query "what is the antenna setup" --limit 3 --format table --client cloud
```

Set `--width 0` to print full chunk contents if you need the entire text. Omit `--client cloud` to inspect a local directory instead.
