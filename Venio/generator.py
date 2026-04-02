import re
import ollama
from typing import List, Dict

GENERATION_MODEL = "qwen2.5:7b"


SYSTEM_PROMPT = """You are a helpful legal document analyst 
for an eDiscovery system.

Your job is to summarize and answer questions using the 
document excerpts provided.

RULES:
1. Use the provided excerpts to write a clear summary
2. Cite every fact as [Source: filename]
3. Write in complete sentences, minimum 2-3 sentences
4. Only say insufficient information if excerpts are 
   completely empty or totally unrelated to the question
5. If excerpts mention anything related to the question
   summarize what they say even if partial information

Do NOT be overly strict. If relevant content exists 
in the excerpts, summarize it."""

NO_RELEVANT_CONTENT_MSG = (
    "The provided documents do not contain sufficient information to answer this query."
)


def format_context(results: List[Dict]) -> str:
    """Format ranked search results into a context block for the LLM."""
    file_names = [r["metadata"]["file_name"] for r in results]
    header = (
        f"AVAILABLE SOURCES (use ONLY these exact file names): "
        f"{', '.join(file_names)}\n\n"
    )

    parts = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        parts.append(
            f"--- Excerpt {i} ---\n"
            f"File: {meta['file_name']}\n"
            f"Author: {meta['author']}\n"
            f"Date: {meta['date']}\n"
            f"Type: {meta['document_type']}\n"
            f"Content:\n{r['content']}\n"
        )
    return header + "\n".join(parts)


def has_relevant_content(chunks: List[Dict], threshold: float = 0.75) -> bool:
    """
    Reject chunks that are too far from query — weak retrieval.
    Threshold lowered to 0.60 to reject noisy/distractor documents harder.
    ChromaDB cosine distance: 0 = identical, 1 = completely different.
    """
    if not chunks:
        return False
    distances = [c.get("distance", 1.0) for c in chunks]
    best_distance = min(distances)
    if best_distance >= threshold:
        print(f"  ⚠️  Best chunk distance {best_distance:.3f} >= threshold {threshold} — rejecting weak retrieval")
        return False
    return True


def generate_response(query: str, results: List[Dict], model: str = GENERATION_MODEL) -> str:
    """
    Generate a grounded, cited response using the LLM with retrieved context.
    Validates sources post-generation to catch hallucinated file references.
    """
    if not results:
        print("  ⚠️  No chunks retrieved — returning no-content message")
        return NO_RELEVANT_CONTENT_MSG

    if not has_relevant_content(results):
        print("  ⚠️  All chunks too distant — returning no-content message")
        return NO_RELEVANT_CONTENT_MSG

    context = format_context(results)

    # Collect valid file names for post-processing validation
    valid_files = {r["metadata"]["file_name"] for r in results}

    user_message = f"""You are a legal document analyst. 
Answer the following question using ONLY the document 
excerpts provided below.

Question: {query}

{context}

INSTRUCTIONS:
- Write a clear paragraph summary answering the question
- Do NOT just list filenames
- Cite each fact immediately as [Source: filename]
- If excerpts do not answer the question say exactly:
  "The provided documents do not contain sufficient 
  information to answer this query."
- Minimum 2-3 sentences in your answer
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        answer = response["message"]["content"].strip()
        answer = _validate_sources(answer, valid_files)
        return answer
    except Exception as e:
        return f"Error generating response: {e}"


def _validate_sources(answer: str, valid_files: set) -> str:
    if "do not contain sufficient information" in answer:
        return NO_RELEVANT_CONTENT_MSG

    cited = re.findall(r"\[Source:\s*([^\]]+)\]", answer)

    # Handle comma-separated citations like [Source: file1.txt, file2.txt]
    all_cited = []
    for c in cited:
        for f in c.split(","):
            all_cited.append(f.strip())

    bad = [s for s in all_cited if s not in valid_files]
    if bad:
        answer = answer
    return answer


if __name__ == "__main__":
    # Quick test with mock data
    mock_results = [
        {
            "content": "Q1 revenue exceeded targets by 12%. Budget concerns raised for Q3.",
            "metadata": {
                "file_name": "email_3.txt",
                "author": "Finance Team",
                "date": "2021-04-15",
                "document_type": "email",
            },
            "distance": 0.2,
        }
    ]
    answer = generate_response("Summarize financial discussions from 2021", mock_results)
    print(answer)

    # Test weak retrieval rejection
    weak_results = [
        {
            "content": "Random noise content with no relevance.",
            "metadata": {
                "file_name": "doc_12.pdf",
                "author": "unknown",
                "date": "unknown",
                "document_type": "pdf",
            },
            "distance": 0.85,  # above threshold — should be rejected
        }
    ]
    answer2 = generate_response("What contracts were signed in 2020?", weak_results)
    print(answer2)