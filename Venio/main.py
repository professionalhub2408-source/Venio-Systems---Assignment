import os
import sys
from ingest import ingest_documents
from chunker import chunk_documents
from vector_store import VectorStore
from intent_parser import parse_intent, build_chroma_filter
from generator import generate_response

DATA_DIR = "venio_dataset"


def build_index(data_dir: str = DATA_DIR, force: bool = False) -> VectorStore:
    """
    Ingest documents, chunk them, and index into ChromaDB.
    Skips re-indexing if index already exists unless --rebuild flag passed.
    """
    store = VectorStore()

    if store.count() > 0 and not force:
        print(f"✅ Vector store already has {store.count()} chunks. Use --rebuild to re-index.")
        return store

    print("\n=== PHASE 1: Data Ingestion ===")
    documents = ingest_documents(data_dir)
    print(f"  ✅ Ingested {len(documents)} documents")

    print("\n=== PHASE 2: Chunking ===")
    chunks = chunk_documents(documents, max_chunk_size=500, overlap=50)
    print(f"  ✅ Created {len(chunks)} chunks")

    print("\n=== PHASE 3: Embedding & Indexing ===")
    store.reset()
    store.index_chunks(chunks)
    print(f"  ✅ Indexed {store.count()} chunks into ChromaDB")

    print("\n✅ Index build complete. Ready for queries.\n")
    return store


def query_pipeline(query: str, store: VectorStore, top_k: int = 5) -> dict:
    """
    Full RAG pipeline for a single query:
    1. Parse intent & extract metadata filters
    2. Build ChromaDB where-filter
    3. Search vector store with filter
    4. Fallback to unfiltered search if no results
    5. Generate grounded cited response
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Step 1: Parse intent
    print("\n--- Step 1: Intent Parsing ---")
    intent = parse_intent(query)
    print(f"  Year:          {intent.get('year')}")
    print(f"  Author:        {intent.get('author')}")
    print(f"  Document type: {intent.get('document_type')}")
    print(f"  Search query:  {intent.get('search_query')}")

    # Step 2: Build metadata filter
    print("\n--- Step 2: Building Metadata Filter ---")
    chroma_filter = build_chroma_filter(intent)
    if chroma_filter:
        print(f"  Filter: {chroma_filter}")
    else:
        print("  No metadata filter — running pure semantic search")

    # Step 3: Search with filter
    print("\n--- Step 3: Retrieval ---")
    search_query = intent.get("search_query", query)
    results = store.search(search_query, n_results=top_k, where_filter=chroma_filter)

    # Step 4: Fallback if no results
    if not results and chroma_filter:
        print("  ⚠️  Filter returned 0 results — falling back to unfiltered search")
        results = store.search(search_query, n_results=top_k)

    if results:
        print(f"  Retrieved {len(results)} chunks:")
        for r in results:
            m = r["metadata"]
            print(
                f"    [{m['file_name']}] "
                f"author={m['author']} | "
                f"date={m['date']} | "
                f"type={m['document_type']} | "
                f"dist={r['distance']:.3f}"
            )
    else:
        print("  ⚠️  No chunks retrieved even after fallback")

    # Step 5: Generate response
    print("\n--- Step 4: Generating Answer ---")
    answer = generate_response(query, results)

    return {
        "query": query,
        "intent": intent,
        "filter": chroma_filter,
        "results": results,
        "answer": answer,
    }


def print_answer(result: dict):
    """Pretty print the final answer."""
    print(f"\n{'='*60}")
    print("ANSWER:")
    print(f"{'='*60}")
    print(result["answer"])
    print(f"{'='*60}\n")


def main():
    """
    Entry point.
    Usage:
      python main.py                          # interactive loop
      python main.py --rebuild                # force re-index then interactive loop
      python main.py "your query here"        # single query mode
      python main.py --rebuild "your query"   # rebuild then single query
    """
    rebuild = "--rebuild" in sys.argv

    # Startup banner
    print("\n" + "="*60)
    print("  Venio Smart — Document Intelligence Pipeline")
    print("  Built by Deepak Balivada | SGP R&D Assignment")
    print("="*60)

    # Build or load index
    store = build_index(DATA_DIR, force=rebuild)

    # Check for single query mode
    query_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if query_args:
        query = " ".join(query_args)
        result = query_pipeline(query, store)
        print_answer(result)
        return

    # Interactive query loop
    print("Type your query below (or 'quit' to exit).")
    print("Example queries:")
    print("  → Find financial discussions from 2021 and summarize them")
    print("  → Show me all emails from John Smith")
    print("  → What contracts were signed in 2020?")
    print("  → Summarize HR policy documents from Sarah Lee")
    print("  → Find all reports about budget planning\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("Exiting. Goodbye.")
            break

        result = query_pipeline(query, store)
        print_answer(result)


if __name__ == "__main__":
    main()