

import re
import json
from typing import Dict, Optional, Tuple

# Try ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

INTENT_MODEL = "llama3.2:1b"


def parse_intent_with_llm(query: str) -> Dict:
    """
    Use Llama 3.2:1b to extract year, author, and document_type filters
    from a natural language query.
    """
    prompt = f"""Extract search filters from the following query. Return ONLY valid JSON with these keys:
- "year": string year like "2021" or null
- "author": author name or null  
- "document_type": one of "email", "report", "contract", "note", "irrelevant" or null
- "search_query": the core semantic search terms (what to actually search for)

Query: "{query}"

JSON:"""

    try:
        response = ollama.chat(
            model=INTENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        text = response["message"]["content"].strip()
        # Extract JSON from response
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "year": parsed.get("year"),
                "author": parsed.get("author"),
                "document_type": parsed.get("document_type"),
                "search_query": parsed.get("search_query", query),
            }
    except Exception as e:
        print(f"LLM intent parsing failed: {e}, falling back to regex")

    return parse_intent_regex(query)


def parse_intent_regex(query: str) -> Dict:
    """Regex-based fallback for extracting filters from a query."""
    result = {
        "year": None,
        "author": None,
        "document_type": None,
        "search_query": query,
    }

    # Extract year
    year_match = re.search(r"\b(20[12]\d)\b", query)
    if year_match:
        result["year"] = year_match.group(1)

    # Compound phrases that should NOT trigger a document_type filter
    compound_phrases = [
        r"meeting\s+notes?", r"release\s+notes?", r"delivery\s+notes?",
    ]
    has_compound = any(re.search(p, query, re.IGNORECASE) for p in compound_phrases)

    type_patterns = {
        "email": r"\bemail[s]?\b",
        "report": r"\breport[s]?\b",
        "contract": r"\bcontract[s]?\b",
        "note": r"\bnote[s]?\b|memo[s]?\b",
    }
    for doc_type, pattern in type_patterns.items():
        if re.search(pattern, query, re.IGNORECASE):
            # Skip "note" filter if it's part of a compound phrase like "meeting notes"
            if doc_type == "note" and has_compound:
                continue
            result["document_type"] = doc_type
            break

    # Extract author
    known_authors = ["John Smith", "Sarah Lee", "Finance Team", "Legal Dept", "HR Team"]
    for author in known_authors:
        if author.lower() in query.lower():
            result["author"] = author
            break

    clean = query
    clean = re.sub(r"\b(from|by|in|during|of|and|the|find|show|get|search|summarize|summary)\b", " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b20[12]\d\b", " ", clean)
    for author in known_authors:
        clean = re.sub(re.escape(author), " ", clean, flags=re.IGNORECASE)
    for pattern in type_patterns.values():
        clean = re.sub(pattern, " ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()
    if clean:
        result["search_query"] = clean

    return result


def parse_intent(query: str) -> Dict:
    """Parse user query to extract intent and metadata filters.
    Uses LLM first, then fills any missing fields with regex fallback.
    """
    llm_result = None
    if OLLAMA_AVAILABLE:
        try:
            ollama.show(INTENT_MODEL)
            llm_result = parse_intent_with_llm(query)
        except Exception:
            pass

    regex_result = parse_intent_regex(query)

    if llm_result is None:
        return regex_result

    # Merge: prefer LLM values, but fill gaps from regex
    merged = {}
    for key in ["year", "author", "document_type", "search_query"]:
        merged[key] = llm_result.get(key) or regex_result.get(key)
    return merged


def build_chroma_filter(intent: Dict) -> Optional[Dict]:
    """
    Convert parsed intent into a ChromaDB where-filter.
    ChromaDB supports $and, $or with field-level $eq, $contains, etc.
    """
    conditions = []

    if intent.get("year"):
       
        try:
            year_int = int(intent["year"])
            conditions.append({"year": {"$eq": year_int}})
        except ValueError:
            pass

    if intent.get("author"):
        conditions.append({"author": {"$eq": intent["author"]}})

    if intent.get("document_type"):
        conditions.append({"document_type": {"$eq": intent["document_type"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


if __name__ == "__main__":
    test_queries = [
        "Find financial discussions from 2021 and summarize them",
        "Show all emails by John Smith",
        "Legal contracts from 2020",
        "What did the Finance Team report in 2021?",
    ]
    for q in test_queries:
        intent = parse_intent(q)
        filt = build_chroma_filter(intent)
        print(f"\nQuery: {q}")
        print(f"  Intent: {intent}")
        print(f"  Filter: {filt}")
