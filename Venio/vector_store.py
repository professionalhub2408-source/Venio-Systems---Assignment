

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

COLLECTION_NAME = "venio_documents"
PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-m3"


class VectorStore:
    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: List[Dict]):
        """Add chunked documents to the vector store."""
        if not chunks:
            return

        batch_size = 64
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start: start + batch_size]
            ids = [c["chunk_id"] for c in batch]
            texts = [c["content"] for c in batch]
            metadatas = []
            for c in batch:
                # Extract year as int for numeric filtering
                year = 0
                if c["date"] and c["date"] != "unknown":
                    try:
                        year = int(c["date"][:4])
                    except (ValueError, IndexError):
                        year = 0
                metadatas.append({
                    "document_id": str(c["document_id"]),
                    "file_name": c["file_name"],
                    "date": c["date"],
                    "year": year,
                    "author": c["author"],
                    "document_type": c["document_type"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                })

            embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        print(f"Indexed {len(chunks)} chunks into ChromaDB")

    def search(
        self,
        query: str,
        n_results: int = 10,
        where_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic search with optional metadata filtering.
        Returns ranked chunks with scores.
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()

        # Always exclude irrelevant documents from search
        base_filter = {"document_type": {"$ne": "irrelevant"}}
        if where_filter:
            combined_filter = {"$and": [base_filter, where_filter]}
        else:
            combined_filter = base_filter

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
            "where": combined_filter,
        }

        results = self.collection.query(**kwargs)

        ranked = []
        for i in range(len(results["ids"][0])):
            ranked.append({
                "chunk_id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return ranked

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


if __name__ == "__main__":
    from ingest import ingest_documents
    from chunker import chunk_documents

    docs = ingest_documents("venio_dataset")
    chunks = chunk_documents(docs)

    store = VectorStore()
    store.reset()
    store.index_chunks(chunks)
    print(f"Total chunks in store: {store.count()}")

    results = store.search("financial discussion revenue", n_results=3)
    for r in results:
        print(f"  [{r['metadata']['file_name']}] dist={r['distance']:.3f}: {r['content'][:60]}...")
