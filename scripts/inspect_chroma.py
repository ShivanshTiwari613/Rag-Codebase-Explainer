"""Utility script to inspect chunks stored in Chroma (local or cloud)."""

import argparse
import os
import textwrap
from pathlib import Path
from typing import Iterable, Optional, Sequence

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DEFAULT_COLLECTION: Optional[str] = None
DEFAULT_PERSIST_DIR = Path("data/chroma")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CLIENT = "cloud"

ENV_API_KEY = os.getenv("CHROMA_API_KEY")
ENV_TENANT = os.getenv("CHROMA_TENANT_ID")
ENV_DATABASE = os.getenv("CHROMA_DATABASE")


def build_cloud_client(api_key: str, tenant: str, database: str) -> chromadb.CloudClient:
    return chromadb.CloudClient(api_key=api_key, tenant=tenant, database=database)


def build_store(
    collection: Optional[str],
    persist_dir: Path,
    model_name: str,
    client_type: str,
    api_key: str | None,
    tenant: str | None,
    database: str | None,
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    if client_type == "cloud":
        missing = [name for name, value in [("collection", collection), ("api_key", api_key), ("tenant", tenant), ("database", database)] if not value]
        if missing:
            raise ValueError(
                "Missing required cloud parameters: " + ", ".join(missing) + ". "
                "Provide them via CLI flags or environment variables."
            )
        client = build_cloud_client(api_key, tenant, database)
        return Chroma(
            client=client,
            collection_name=collection,
            embedding_function=embeddings,
        )

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Persist directory '{persist_dir}' does not exist. Ingest documents first or choose --client cloud."
        )

    if not collection:
        collection = "codebase_embeddings"

    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


def fmt_metadata(metadata: dict) -> str:
    if not metadata:
        return "{}"
    items = ", ".join(f"{k}={metadata[k]!r}" for k in sorted(metadata))
    return f"{{{items}}}"


def preview(text: str, width: int) -> str:
    if width is None:
        return text
    return textwrap.shorten(text.replace("\n", " "), width=width)


def print_table(rows: Iterable[Sequence[str]], headers: Sequence[str]) -> None:
    rows = list(rows)
    if not rows:
        print("(no rows)")
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))


def list_cloud_collections(api_key: str, tenant: str, database: str) -> None:
    client = build_cloud_client(api_key, tenant, database)
    rows = [(c.name, str(c.metadata or {}), str(c.dimension)) for c in client.list_collections()]
    print_table(rows, headers=("Name", "Metadata", "Dimension"))


def list_chunks(store: Chroma, limit: int, width: int, as_table: bool) -> None:
    data = store.get(include=["documents", "metadatas"], limit=limit)
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        print("No chunks found in the collection.")
        return

    if as_table:
        rows = []
        for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            meta_copy = dict(meta or {})
            source = str(meta_copy.pop("source", "unknown"))
            rows.append(
                (
                    str(idx),
                    source,
                    fmt_metadata(meta_copy),
                    preview(doc, width),
                )
            )
        print_table(rows, headers=("Index", "Source", "Metadata", "Content Preview"))
    else:
        for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            print(f"Chunk {idx}")
            print(f"  Metadata: {fmt_metadata(meta)}")
            print(f"  Content: {preview(doc, width)}")
            print("-" * 80)


def similarity_search(store: Chroma, query: str, limit: int, width: int, as_table: bool) -> None:
    results = store.similarity_search(query, k=limit)
    if not results:
        print("No results for similarity search. Try a different query.")
        return

    if as_table:
        rows = []
        for idx, doc in enumerate(results, start=1):
            metadata = dict(doc.metadata or {})
            score = str(metadata.pop("score", "n/a"))
            source = str(metadata.pop("source", "unknown"))
            rows.append(
                (
                    str(idx),
                    score,
                    source,
                    fmt_metadata(metadata),
                    preview(doc.page_content, width),
                )
            )
        print_table(rows, headers=("Rank", "Score", "Source", "Metadata", "Content Preview"))
    else:
        for idx, doc in enumerate(results, start=1):
            print(f"Result {idx}")
            print(f"  Score: {doc.metadata.get('score', 'n/a')}")
            print(f"  Source: {doc.metadata.get('source', 'unknown')}")
            print(f"  Metadata: {fmt_metadata(doc.metadata)}")
            print(f"  Content: {preview(doc.page_content, width)}")
            print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect chunks stored in Chroma.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name to inspect.")
    parser.add_argument(
        "--persist-dir",
        default=str(DEFAULT_PERSIST_DIR),
        help="Directory containing the persisted Chroma data (local client).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="SentenceTransformer model name used for embeddings.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of chunks or search results to display.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=200,
        help="Character width for content previews. Use 0 to show full chunks.",
    )
    parser.add_argument(
        "--query",
        help="If supplied, run a similarity search for the query instead of listing all chunks.",
    )
    parser.add_argument(
        "--format",
        choices=["list", "table"],
        default="list",
        help="Output format for displayed chunks.",
    )
    parser.add_argument(
        "--client",
        choices=["local", "cloud"],
        default=DEFAULT_CLIENT,
        help="Use local persistence or connect to Chroma Cloud.",
    )
    parser.add_argument("--api-key", default=ENV_API_KEY, help="Chroma Cloud API key (use with --client cloud).")
    parser.add_argument("--tenant", default=ENV_TENANT, help="Chroma Cloud tenant ID (use with --client cloud).")
    parser.add_argument("--database", default=ENV_DATABASE, help="Chroma Cloud database name (use with --client cloud).")
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List available collections instead of reading documents.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width = None if args.width == 0 else args.width

    if args.list_collections:
        if args.client != "cloud":
            raise ValueError("--list-collections is only supported for the cloud client.")
        missing = [name for name, value in [("api_key", args.api_key), ("tenant", args.tenant), ("database", args.database)] if not value]
        if missing:
            raise ValueError("Missing values: " + ", ".join(missing))
        list_cloud_collections(args.api_key, args.tenant, args.database)
        return

    store = build_store(
        collection=args.collection,
        persist_dir=Path(args.persist_dir),
        model_name=args.model,
        client_type=args.client,
        api_key=args.api_key,
        tenant=args.tenant,
        database=args.database,
    )
    as_table = args.format == "table"

    if args.query:
        similarity_search(store, args.query, args.limit, width, as_table)
    else:
        list_chunks(store, args.limit, width, as_table)


if __name__ == "__main__":
    main()
