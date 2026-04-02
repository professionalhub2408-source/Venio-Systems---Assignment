"""
Venio Smart - Main Orchestrator (main.py)
Ties together ingestion, chunking, vector search, intent parsing, and generation.
Provides both CLI and programmatic interfaces.
"""

import os
import sys
from ingest import ingest_documents
from chunker import chunk_documents
from vector_store import VectorStore
from intent_parser import parse_intent, build_chroma_filter
from generator import generate_response

DATA_DIR = "venio_dataset"


def build_index(data_dir: str = DATA_DIR, force: bool = False) -> VectorStore:
    """Ingest documents, chunk them, and index into ChromaDB."""
    store = VectorStore()

    if store.count() > 0 and not force:
        print(f"Vector store already has {store.count()} chunks. Use --rebuild to re-index.")
        return store

    print("=== Data Ingestion & Pre-processing ===")
    documents = ingest_documents(data_dir)
    print(f"  Ingested {len(documents)} documents")

    print("=== Chunking ===")
    chunks = chunk_documents(documents, max_chunk_size=500, overlap=50)
    print(f"  Created {len(chunks)} chunks")

    print("=== Embedding & Indexing ===")
    store.reset()
    store.index_chunks(chunks)
    print(f"  Indexed {store.count()} chunks into ChromaDB")

    return store


def query_pipeline(query: str, store: VectorStore, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
    1. Parse intent & extract filters
    2. Search vector store with filters
    3. Generate grounded response
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Step 1: Intent Parsing
    print("\n--- Intent Parsing ---")
    intent = parse_intent(query)
    print(f"  Parsed intent: {intent}")

    # Step 2: Build metadata filter
    chroma_filter = build_chroma_filter(intent)
    if chroma_filter:
        print(f"  Metadata filter: {chroma_filter}")

    # Step 3: Semantic search with filtering
    print("\n--- Retrieval ---")
    search_query = intent.get("search_query", query)
    results = store.search(search_query, n_results=top_k, where_filter=chroma_filter)

    if not results and chroma_filter:
        # Fallback: try without filter
        print("  No results with filter, retrying without metadata filter...")
        results = store.search(search_query, n_results=top_k)

    print(f"  Retrieved {len(results)} chunks:")
    for r in results:
        m = r["metadata"]
        print(f"    [{m['file_name']}] {m['author']} ({m['date']}) dist={r['distance']:.3f}")

    # Step 4: Generate response
    print("\n--- Generation ---")
    answer = generate_response(query, results)

    return {
        "query": query,
        "intent": intent,
        "filter": chroma_filter,
        "results": results,
        "answer": answer,
    }


def main():
    # Parse CLI args
    rebuild = "--rebuild" in sys.argv

    # Build / load index
    store = build_index(DATA_DIR, force=rebuild)

    # Interactive query loop if no query arg
    query_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if query_args:
        query = " ".join(query_args)
        result = query_pipeline(query, store)
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(f"{'='*60}")
        print(result["answer"])
    else:
        print("\n--- Venio Smart: Document Intelligence Pipeline ---")
        print("Type your query (or 'quit' to exit):\n")
        while True:
            try:
                query = input("Query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            result = query_pipeline(query, store)
            print(f"\n{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(result["answer"])
            print()


if __name__ == "__main__":
    main()
