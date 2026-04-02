

import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """Sanitize text: remove known noise patterns, normalize whitespace."""
    # Remove known noise patterns
    text = re.sub(r'---\s*PAGE BREAK\s*---', '', text)
    text = re.sub(r'\$\$\$.*?\$\$\$', '', text)
    text = re.sub(r'#{3,}', '', text)
    text = re.sub(r'\?{2,}', '', text)
    text = re.sub(r'!{2,}', '', text)
    # Collapse whitespace
    text = re.sub(r"[^\S\n]+", " ", text)   # collapse spaces (keep newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 newlines
    return text.strip()


def is_meaningful(text: str, min_length: int = 100) -> bool:
    """Check if text has enough real alphabetic content to be useful."""
    alpha_only = re.sub(r'[^a-zA-Z\s]', '', text)
    return len(alpha_only.strip()) >= min_length


def extract_email_body(text: str) -> str:
    """Strip email headers, return just the body."""
    lines = text.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if re.match(r"^(From|To|Date|Subject):", line):
            body_start = i + 1
            continue
        if line.strip() == "" and body_start > 0:
            body_start = i + 1
            break
    return "\n".join(lines[body_start:]).strip()


def semantic_chunk(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks using paragraph boundaries first, then sentence
    boundaries, with overlap for context continuity.
    """
    text = clean_text(text)
    if len(text) <= max_chunk_size:
        return [text]

    # Split on paragraph boundaries
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If single paragraph exceeds limit, split by sentences
        if len(para) > max_chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= max_chunk_size:
                    current_chunk = (current_chunk + " " + sent).strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sent
        elif len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    # Add overlap between consecutive chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return chunks


def chunk_documents(documents: List[Dict], max_chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Take ingested documents and return chunked records with metadata preserved.
    """
    all_chunks = []
    for doc in documents:
        content = doc["content"]
        # For emails, chunk just the body
        if doc["document_type"] == "email":
            content = extract_email_body(content)

        content = clean_text(content)

        # Skip documents with insufficient meaningful content
        if not is_meaningful(content):
            continue

        chunks = semantic_chunk(content, max_chunk_size, overlap)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['document_id']}_chunk_{i}",
                "document_id": doc["document_id"],
                "file_name": doc["file_name"],
                "content": chunk_text,
                "date": doc["date"],
                "author": doc["author"],
                "document_type": doc["document_type"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    return all_chunks


if __name__ == "__main__":
    from ingest import ingest_documents
    docs = ingest_documents("venio_dataset")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    for c in chunks[:5]:
        print(f"  {c['chunk_id']}: {c['content'][:60]}...")
