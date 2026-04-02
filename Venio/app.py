"""
Venio Smart - Streamlit UI
Interactive web interface for the document intelligence pipeline.
"""

import streamlit as st
from main import build_index, query_pipeline

st.set_page_config(page_title="Venio Smart", page_icon="🔍", layout="wide")
st.title("Venio Smart: Document Intelligence Pipeline")
st.caption("Modular RAG system for eDiscovery — semantic search with metadata filtering")

# Initialize vector store (cached)
@st.cache_resource
def get_store():
    return build_index("venio_dataset", force=False)

store = get_store()
st.sidebar.success(f"Vector store loaded: {store.count()} chunks indexed")

# Sidebar: metadata filter overrides
st.sidebar.header("Manual Filters (optional)")
year_filter = st.sidebar.selectbox("Year", [None, "2020", "2021", "2022"])
author_filter = st.sidebar.selectbox("Author", [None, "John Smith", "Sarah Lee", "Finance Team", "Legal Dept", "HR Team"])
type_filter = st.sidebar.selectbox("Document Type", [None, "email", "report", "contract", "note"])
top_k = st.sidebar.slider("Top K results", 3, 15, 5)

# Query input
query = st.text_input("Enter your query:", placeholder="e.g. Find financial discussions from 2021 and summarize them")

if query:
    with st.spinner("Processing query..."):
        result = query_pipeline(query, store, top_k=top_k)

    # Show parsed intent
    with st.expander("Parsed Intent & Filters", expanded=False):
        st.json(result["intent"])
        if result["filter"]:
            st.json(result["filter"])

    # Show answer
    st.subheader("Answer")
    st.markdown(result["answer"])

    # Show source documents
    st.subheader("Source Documents")
    for i, r in enumerate(result["results"], 1):
        m = r["metadata"]
        with st.expander(f"[{i}] {m['file_name']} — {m['author']} ({m['date']}) | distance: {r['distance']:.3f}"):
            st.text(r["content"])

# Sample queries
st.sidebar.header("Sample Queries")
samples = [
    "Find financial discussions from 2021 and summarize them",
    "Show all emails by John Smith",
    "What legal contracts were created in 2020?",
    "Summarize Finance Team reports from 2021",
    "HR policy updates",
]
for s in samples:
    if st.sidebar.button(s, key=s):
        st.session_state["query"] = s
        st.rerun()
