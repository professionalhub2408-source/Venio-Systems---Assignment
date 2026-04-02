"""
Venio Smart - Document Ingestion Pipeline
Reads .txt, .pdf files and metadata.csv, returns structured document records.
"""

import os
import csv
import re
from typing import List, Dict, Optional
from PyPDF2 import PdfReader


def parse_email_headers(text: str) -> Dict[str, str]:
    """Extract From, To, Date, Subject from email-formatted text files."""
    headers = {}
    for field in ["From", "To", "Date", "Subject"]:
        match = re.search(rf"^{field}:\s*(.+)$", text, re.MULTILINE)
        if match:
            headers[field.lower()] = match.group(1).strip()
    return headers


def read_txt_file(filepath: str) -> str:
    """Read a plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def read_pdf_file(filepath: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    """Load metadata.csv into a dict keyed by file_name."""
    metadata = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["file_name"]] = {
                "document_id": row["document_id"],
                "date": row["date"].split(" ")[0],  # keep YYYY-MM-DD
                "author": row["author"],
                "document_type": row["document_type"],
            }
    return metadata


def ingest_documents(data_dir: str) -> List[Dict]:
    """
    Ingest all documents from data_dir/documents/ with metadata from data_dir/metadata.csv.
    Returns a list of document records.
    """
    docs_dir = os.path.join(data_dir, "documents")
    metadata_path = os.path.join(data_dir, "metadata.csv")

    metadata = load_metadata(metadata_path)
    documents = []

    for fname in sorted(os.listdir(docs_dir)):
        fpath = os.path.join(docs_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # Read content based on file type
        if fname.endswith(".pdf"):
            content = read_pdf_file(fpath)
        elif fname.endswith(".txt"):
            content = read_txt_file(fpath)
        else:
            continue

        if not content:
            continue

        # Get metadata (fall back to defaults if missing)
        meta = metadata.get(fname, {})
        email_headers = {}
        if meta.get("document_type") == "email" and fname.endswith(".txt"):
            email_headers = parse_email_headers(content)

        # Determine document_type with override rules:
        # Rule 1: "irrelevant*" files → irrelevant
        # Rule 2: ALL .pdf files → irrelevant (noise in this dataset)
        # Rule 3: "email*.txt" files → email
        # Rule 4: Any other .txt → txt
        if fname.startswith("irrelevant"):
            doc_type = "irrelevant"
        elif fname.endswith(".pdf"):
            doc_type = "irrelevant"
        elif fname.startswith("email") and fname.endswith(".txt"):
            doc_type = "email"
        elif fname.endswith(".txt"):
            doc_type = "txt"
        else:
            doc_type = meta.get("document_type", "unknown")

        doc = {
            "document_id": meta.get("document_id", fname),
            "file_name": fname,
            "content": content,
            "date": meta.get("date", email_headers.get("date", "unknown")),
            "author": meta.get("author", email_headers.get("from", "unknown")),
            "document_type": doc_type,
        }
        documents.append(doc)

    return documents


if __name__ == "__main__":
    docs = ingest_documents("venio_dataset")
    print(f"Ingested {len(docs)} documents")
    for d in docs[:3]:
        print(f"  [{d['document_id']}] {d['file_name']} | {d['document_type']} | {d['author']} | {d['date']}")
        print(f"    Content preview: {d['content'][:80]}...")
