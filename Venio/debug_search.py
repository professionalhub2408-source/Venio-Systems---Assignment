from vector_store import VectorStore

store = VectorStore()
print(f"Total chunks in store: {store.count()}")

results = store.search("meeting notes on budget planning", n_results=10)
print()
for r in results:
    m = r["metadata"]
    print(f"[{m['file_name']}] author={m['author']} date={m['date']} type={m['document_type']} dist={r['distance']:.4f}")
    print(f"  Content: {r['content'][:120]}...")
    print()
