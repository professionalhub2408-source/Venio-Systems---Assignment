"""
Venio Smart - RAG Response Generator
Uses Qwen 2.5:7b via Ollama with a strict auditor prompt for grounded, cited summaries.
"""

import ollama
from typing import List, Dict

GENERATION_MODEL = "qwen2.5:7b"


SYSTEM_PROMPT = """You are a strict document auditor for an eDiscovery system. 
Your job is to answer the user's question based ONLY on the provided document excerpts.

Rules:
1. ONLY use information from the provided excerpts. Do NOT add external knowledge.
2. For every claim, cite the source file in brackets like [Source: filename.txt].
3. If the excerpts do not contain enough information, say so explicitly.
4. Be concise and factual. Organize the response clearly.
5. If asked to summarize, provide a structured summary with key points."""


def format_context(results: List[Dict]) -> str:
    """Format ranked search results into a context block for the LLM."""
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
    return "\n".join(parts)


def generate_response(query: str, results: List[Dict], model: str = GENERATION_MODEL) -> str:
    """
    Generate a grounded, cited response using the LLM with retrieved context.
    """
    if not results:
        return "No relevant documents found for your query."

    context = format_context(results)

    user_message = f"""Based on the following document excerpts, answer this question:

Question: {query}

{context}

Provide a clear, cited answer based ONLY on the excerpts above."""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Error generating response: {e}"


if __name__ == "__main__":
    # Quick test with mock data
    mock_results = [
        {
            "content": "This is a discussion regarding financial discussion about Q2 revenue.",
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
