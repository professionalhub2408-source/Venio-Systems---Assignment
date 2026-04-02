# Venio Smart — Document Intelligence Pipeline
> Built by Deepak Balivada 

---

## What This System Does

eDiscovery is the process of finding legally relevant documents inside massive collections — emails, contracts, reports, PDFs — during investigations or litigation. The problem is scale and precision. You cannot manually read 10,000 files. You need a system that understands *what you're looking for*, not just what words you typed.

**Venio Smart** ingests raw documents, understands their content and metadata, and lets users ask natural language questions like *"Find financial discussions from 2021 by John Smith"* — returning cited, grounded answers pulled directly from the source documents. No hallucinated summaries. No guessing. Every claim traced back to a file.

---

## System Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        INDEX PIPELINE  (runs once)                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  venio_dataset/documents/                                                ║
║  ├── email_0.txt ... email_9.txt   (structured email format)            ║
║  ├── doc_10.pdf  ... doc_19.pdf    (noisy PDF reports/contracts)        ║
║  └── irrelevant_20.txt ... _29.txt (noise/distractor documents)         ║
║                    │                                                     ║
║                    ▼                                                     ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  ingest.py                                          │                ║
║  │  • read_pdf()     → PyPDF2, page-by-page            │                ║
║  │  • read_txt()     → UTF-8, errors="replace"         │                ║
║  │  • parse_headers()→ Regex: From/To/Date/Subject     │                ║
║  │  • load_metadata()→ metadata.csv → {id,date,author} │                ║
║  └──────────────────────────┬──────────────────────────┘                ║
║                             ▼                                            ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  chunker.py  (custom semantic chunking)             │                ║
║  │  • clean_text()         → collapse whitespace noise │                ║
║  │  • extract_email_body() → strip header pollution    │                ║
║  │  • semantic_chunk()     → paragraph → sentence      │                ║
║  │    max=500 chars | overlap=50 chars                 │                ║
║  └──────────────────────────┬──────────────────────────┘                ║
║                             ▼                                            ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  vector_store.py                                    │                ║
║  │  • BAAI/bge-m3 → 768-dim embeddings                │                ║
║  │  • normalize_embeddings=True (L2 normalization)     │                ║
║  │  • batch_size=64                                    │                ║
║  │  • ChromaDB.upsert() → HNSW cosine index on disk   │                ║
║  └──────────────────────────┬──────────────────────────┘                ║
║                             ▼                                            ║
║              chroma_db/  (SQLite + HNSW segments)                       ║
║              30 chunks persisted — survives restarts                     ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                      QUERY PIPELINE  (runs per query)                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  User: "Find financial discussions from 2021 by John Smith"             ║
║                    │                                                     ║
║                    ▼                                                     ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  intent_parser.py                                   │                ║
║  │                                                     │                ║
║  │  Layer 1: Llama 3.2:1b (via Ollama, local)         │                ║
║  │  → Prompt: "Extract year, author, doc_type as JSON" │                ║
║  │  → Output: {year:"2021", author:"John Smith",       │                ║
║  │             type:"email", search_query:"financial"} │                ║
║  │                                                     │                ║
║  │  Layer 2: Regex fallback (if LLM fails/unavailable) │                ║
║  │  → \b(20[12]\d)\b for year                         │                ║
║  │  → compound phrase guard ("meeting notes" ≠ "note") │                ║
║  │                                                     │                ║
║  │  Merge: prefer LLM values, fill gaps from regex    │                ║
║  └──────────────────────────┬──────────────────────────┘                ║
║                             ▼                                            ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  build_chroma_filter()                              │                ║
║  │  {$and: [                                           │                ║
║  │    {year: {$eq: 2021}},                             │                ║
║  │    {author: {$eq: "John Smith"}},                   │                ║
║  │    {document_type: {$eq: "email"}}                  │                ║
║  │  ]}                                                 │                ║
║  └──────────────────────────┬──────────────────────────┘                ║
║                             ▼                                            ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  vector_store.search()                              │                ║
║  │  • BGE-M3 encodes query → 768-dim vector            │                ║
║  │  • ChromaDB: apply where-filter FIRST               │                ║
║  │  • Then cosine similarity rank within filtered set  │                ║
║  │  • Returns top-K chunks + metadata + distance       │                ║
║  │                                                     │                ║
║  │  Fallback: if 0 results → retry without filter      │                ║
║  └──────────────────────────┬──────────────────────────┘                ║
║                             ▼                                            ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  generator.py                                       │                ║
║  │  • format_context() → numbered excerpt blocks       │                ║
║  │    [1] file: email_3.txt | author: John Smith       │                ║
║  │        date: 2021-03-15  | type: email              │                ║
║  │        content: "Q1 revenue exceeded..."            │                ║
║  │                                                     │                ║
║  │  • Qwen 2.5:7b (via Ollama, local)                 │                ║
║  │    temperature=0.1 | num_ctx=4096                   │                ║
║  │    System: "You are a document auditor.             │                ║
║  │    ONLY use provided excerpts. Cite every           │                ║
║  │    claim as [Source: filename]."                    │                ║
║  └──────────────────────────┬──────────────────────────┘                ║
║                             ▼                                            ║
║  Cited Answer: "John Smith discussed Q1 revenue exceeding targets       ║
║  [Source: email_3.txt] and flagged budget concerns for Q3               ║
║  [Source: email_7.txt]."                                                ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════╗
║                            INTERFACES                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   main.py (CLI)          app.py (Streamlit UI)     api.py (FastAPI)     ║
║   python main.py         streamlit run app.py      uvicorn api:app      ║
║   --rebuild flag         sidebar filter override   POST /query          ║
║   interactive loop       top-K slider              GET  /health         ║
║                          source doc expanders      JSON in/out          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Why I Chose This Architecture — Research & Design Decisions

### 1. Embedding Model — BGE-M3 over OpenAI

Most RAG tutorials default to OpenAI's embedding API. I didn't, for two reasons.

**Reason 1 — Privacy.** This is an eDiscovery system. Legal documents contain privileged communications, PII, financial records. Sending that data to an external API is a liability. BGE-M3 runs fully local — nothing leaves the machine.

**Reason 2 — Hybrid Retrieval capability.** BGE-M3 supports both dense (semantic) and sparse (keyword) retrieval. This matters specifically for legal document search. A query like *"find hostile tone in communications"* needs semantic understanding. A query like *"find document ID C-2021-447"* needs exact keyword matching. OpenAI's `text-embedding-3` is dense-only — it would fail on the second type of query. BGE-M3 handles both.

| | BGE-M3 | OpenAI text-embedding-3-small |
|---|---|---|
| Runs locally | ✅ | ❌ (API only) |
| Dense retrieval | ✅ | ✅ |
| Sparse/keyword retrieval | ✅ | ❌ |
| Multilingual | ✅ | ✅ |
| Data privacy | ✅ | ❌ |
| Cost at scale | Free | $0.02/1M tokens |

---

### 2. Vector Database — ChromaDB over FAISS

I looked at both FAISS and ChromaDB. FAISS is faster in raw vector search, but it has a critical limitation for this use case: **FAISS has no native metadata filtering.** To filter by year + author + document type before semantic ranking, you'd have to load all vectors, run the search, then apply Python-side filtering — which means irrelevant documents compete in the similarity ranking before you filter them out.

ChromaDB's where-clause runs the metadata filter **before** the vector search, so BGE-M3 only compares embeddings within the correct subset. For eDiscovery where you almost always know the date range or document type you care about, this is the right design.

| | ChromaDB | FAISS |
|---|---|---|
| Metadata filtering | ✅ Native where-clause | ❌ Post-search Python filter |
| Persistent storage | ✅ SQLite on disk | ❌ Requires manual save/load |
| Filter-then-search | ✅ | ❌ |
| Raw ANN speed | Good | Faster |
| Production scale (>1M docs) | Needs migration to Qdrant | Needs migration to custom infra |

---

### 3. Dual LLM Design — Llama 3.2:1b + Qwen 2.5:7b

This was my own design decision, driven by a real hardware constraint. I'm running this on an RTX 2050 — a mid-range laptop GPU. Running a 7B model for every part of the pipeline was slow enough to break the interactive experience.

So I split the work by cognitive load:

- **Intent parsing is a structured extraction task.** The input is a short query. The output is a JSON object with 4 fields. A 1B model can do this reliably and runs in under a second.
- **Response generation is a reasoning task.** The model needs to read 5 document excerpts, cross-reference them, synthesize an answer, and cite every claim. This needs a larger model.

Running Llama 3.2:1b for parsing is roughly 5x faster than using Qwen 2.5:7b for the same job. The 7B model only runs when it's actually needed — for the answer. This gives the experience of a fast, responsive system on consumer hardware.

| Model | Parameters | Job | Temperature | Latency |
|---|---|---|---|---|
| Llama 3.2:1b | 1B | Intent parsing → JSON filters | 0.0 | ~0.8s |
| Qwen 2.5:7b | 7B | RAG answer generation + citations | 0.1 | ~4-6s |

Both run via **Ollama** — a local model runner. No API keys. No cloud dependency. No data ever leaves the machine. For eDiscovery, where document confidentiality is a legal requirement, this is not a nice-to-have, it's a necessity.

---

### 4. Chunking Strategy — Paragraph-first Semantic Chunking

Fixed-size token chunking (the default in LangChain and LlamaIndex) splits text every N tokens regardless of where a sentence or paragraph ends. For general web text this is acceptable. For legal documents it is not — a legal argument or financial summary broken mid-sentence loses its meaning and retrieves incorrectly.

My chunking hierarchy:
1. **Paragraph boundaries first** (`\n\n+`) — legal and financial documents are structured argumentatively. Each paragraph is one logical unit.
2. **Sentence boundaries second** (`(?<=[.!?])\s+`) — for paragraphs longer than 500 characters, split at sentence ends rather than mid-word.
3. **50-character overlap** between consecutive chunks — prevents information loss at boundaries where a key fact might span two chunks.
4. **Email body extraction** — for email documents, strip `From/To/Date/Subject` headers before chunking. These headers carry no semantic content for embedding but inflate chunk space. Headers are preserved as structured metadata instead.

---

### 5. Noise Handling

The dataset includes intentionally noisy documents with artifacts like `$$$ RANDOM HEADER $$$` and `--- PAGE BREAK ---`. Two-pass `clean_text()` handles this:
- Pass 1: `[^\S\n]+` → collapses all horizontal whitespace to single space
- Pass 2: `\n{3,}` → caps consecutive newlines at 2

`read_txt_file()` uses `errors="replace"` so malformed encoding characters never crash the pipeline. PyPDF2 skips pages that return `None` (blank or unreadable pages in scanned PDFs).

---

### 6. Anti-Hallucination Design

Three layers:

1. **Strict system prompt** — the generator is told it is a document auditor. Rules: only use provided excerpts, cite every claim with `[Source: filename]`, explicitly state when information is insufficient.
2. **Low temperature (0.1)** — near-deterministic output. The model doesn't invent creative answers.
3. **Fallback transparency** — if the filter returns 0 results and the system falls back to unfiltered search, it still generates from real retrieved chunks, not from model memory.

**Where hallucinations can still occur and how to fix them:**

| Risk point | What happens | Fix |
|---|---|---|
| Weak retrieval | Retrieved chunks are only tangentially related. LLM fills gaps. | Add distance threshold cutoff — reject chunks with cosine distance > 0.7 |
| Citation drift | LLM attributes a fact to the wrong source file | Structured output with per-sentence citation enforcement |
| Fallback search | Noise documents retrieved when filter fails | Document classification to exclude `irrelevant` type at generation time |
| Context overflow | num_ctx=4096 truncates long excerpts silently | Truncate chunks at format time, not inside the model |

---

## Challenges Faced — Real Problems, Real Fixes

These are the actual problems I hit while building this system, not theoretical ones.

---

### Challenge 1: ChromaDB Metadata Filter Returning Zero Results

**What happened:**
My first queries kept returning zero results even though I could see the documents were indexed. The metadata filter was silently failing.

**Root cause:**
ChromaDB's where-clause requires exact type matching. My metadata stored year as a string `"2021"` but my filter was querying it as an integer `{year: {$eq: 2021}}`. ChromaDB didn't throw an error — it just returned nothing.

**Fix:**
Standardized all metadata to strings at ingest time. Year stored as `"2021"`, filter uses `{$eq: "2021"}`. Added a schema validation step in `vector_store.py` that enforces types before upsert.

**Lesson:** Always validate metadata types before storage. Silent type mismatches in vector databases are hard to debug.

---

### Challenge 2: LLM Intent Parser Returning Malformed JSON

**What happened:**
Llama 3.2:1b would sometimes return JSON wrapped in markdown code blocks like ` ```json {...} ``` ` instead of raw JSON. `json.loads()` would crash the pipeline.

**Fix:**
Added a JSON extraction wrapper that strips markdown fences before parsing:
```python
def extract_json(text):
    text = re.sub(r"```json|```", "", text).strip()
    return json.loads(text)
```
Also added a full try/except fallback that switches to regex-based extraction if JSON parsing fails entirely. The dual-layer design (LLM + regex fallback) means the pipeline never crashes due to LLM formatting errors.

---

### Challenge 3: Email Headers Polluting Embeddings

**What happened:**
Early retrieval results were dominated by header matches. Query `"financial discussions"` would retrieve emails because `From: finance@company.com` was embedded as part of the chunk content — the word "finance" in an email address was matching semantically.

**Fix:**
Built `extract_email_body()` in `chunker.py` that detects structured header blocks using regex (`^From:`, `^To:`, `^Date:`, `^Subject:`) and strips them from chunk content before embedding. Headers are preserved separately as structured metadata fields — so they're still searchable through the filter layer, just not polluting the semantic embedding space.

---

### Challenge 4: BGE-M3 First-Load Latency

**What happened:**
BGE-M3 is a large model (~570MB). First load from disk took 8-12 seconds, which made the system feel broken on first query.

**Fix:**
Moved embedding model initialization to startup time rather than per-query time. The model loads once when the pipeline starts and stays in memory. Added a loading indicator in the Streamlit UI so users know the system is warming up rather than hanging.

---

### Challenge 5: Noisy PDF Pages Breaking Chunking

**What happened:**
Several PDFs in the dataset had pages that returned `None` from PyPDF2 — either blank pages or scanned images with no extractable text. These `None` values would crash `clean_text()` downstream.

**Fix:**
Added explicit None-check in `read_pdf()`:
```python
for page in pdf.pages:
    text = page.extract_text()
    if text:  # skip None and empty pages
        pages.append(text)
```
Also added `errors="replace"` to all text file reads to handle malformed UTF-8 encoding without crashing.

---

### Challenge 6: Qwen 2.5:7b Slow on First Generation

**What happened:**
On a mid-range RTX 2050 GPU, Qwen 2.5:7b takes 4-6 seconds per generation. For an interactive CLI this feels slow. Running it for intent parsing too made every query take 10+ seconds.

**Fix:**
This is the core reason for the dual LLM design. Routing the structured extraction task (intent parsing) to Llama 3.2:1b (1B params, ~0.8s) and only using Qwen 2.5:7b for the final answer generation cut total latency roughly in half on consumer hardware.

---

## What Would Break at 1 Million Documents

| Component | Current limit | Production fix |
|---|---|---|
| ChromaDB HNSW | Loads index into memory — OOMs well before 1M docs | Migrate to Qdrant with disk-based HNSW |
| Single-process embedding | Sequential batch=64 — hours for 1M docs | Async worker queue (Celery + Ray) with GPU parallelism |
| Regex author matching | Hardcoded name list — breaks on "J. Smith", "Smith J." | spaCy NER-based entity extraction at ingest time |
| Ollama local inference | Single-threaded, no concurrent requests | vLLM for batched inference or cloud endpoint with rate limiting |
| Year-only date filter | `{year: {$eq: 2021}}` — can't do date ranges | Store date as ISO string, use `$gte`/`$lte` range operators |
| No audit log | No record of what was queried or returned | Append-only query log: timestamp, query, filters applied, doc IDs returned |

---

## Project Structure

```
Venio/
├── ingest.py          # Document reading + metadata loading
├── chunker.py         # Semantic chunking pipeline
├── vector_store.py    # BGE-M3 embeddings + ChromaDB HNSW index
├── intent_parser.py   # Dual-layer query understanding (LLM + regex)
├── generator.py       # Qwen 2.5:7b RAG answer generation
├── main.py            # Pipeline orchestrator (CLI)
├── app.py             # Streamlit web UI
├── api.py             # FastAPI REST API
├── requirements.txt
└── venio_dataset/
    ├── metadata.csv
    └── documents/
        ├── email_0.txt ... email_9.txt
        ├── doc_10.pdf  ... doc_19.pdf
        └── irrelevant_20.txt ... irrelevant_29.txt
```

---

## Setup & Demo

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Pull the two models:
```bash
ollama pull llama3.2:1b
ollama pull qwen2.5:7b
```

### Install
```bash
git clone <repo>
cd Venio
pip install -r requirements.txt
```

### Run — Three Ways

**CLI (interactive)**
```bash
python main.py
# Query> Find financial discussions from 2021
```

**Web UI**
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

**REST API**
```bash
uvicorn api:app --reload
# Swagger UI at http://127.0.0.1:8000/docs
```

**API example:**
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find financial discussions from 2021", "top_k": 5}'
```

**Force re-index:**
```bash
python main.py --rebuild
```

---

## Sample Queries

```
Find financial discussions from 2021 and summarize them
Show me all emails from John Smith
What contracts were signed in 2020?
Summarize HR policy documents from Sarah Lee
Find all reports about budget planning
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Embeddings | BAAI/bge-m3 | Local, hybrid dense+sparse, privacy-safe |
| Vector DB | ChromaDB | Native metadata filtering before search |
| Intent Parsing | Llama 3.2:1b via Ollama | Fast, local, structured JSON extraction |
| Answer Generation | Qwen 2.5:7b via Ollama | Strong reasoning, citation-aware |
| PDF Parsing | PyPDF2 | Lightweight, page-level control |
| Web UI | Streamlit | Fast prototyping, interactive filters |
| REST API | FastAPI | Production-ready, auto Swagger docs |
| Chunking | Custom semantic chunker | Paragraph-first, legal-document-aware |
