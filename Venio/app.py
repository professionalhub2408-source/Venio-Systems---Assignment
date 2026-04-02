
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

    # SECTION 1 — Intent & Filters
    with st.expander("Parsed Intent & Filters", expanded=False):
        intent = result["intent"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Year", intent.get("year") or "—")
        col2.metric("Author", intent.get("author") or "—")
        col3.metric("Document Type", intent.get("document_type") or "—")
        col4.metric("Search Query", intent.get("search_query") or "—")
        st.markdown("**ChromaDB Filter Applied:**")
        if result["filter"]:
            st.json(result["filter"])
        else:
            st.info("No metadata filter applied")

    # SECTION 2 — Answer
    st.subheader("Answer")
    answer = result["answer"]
    if "do not contain sufficient information" in answer:
        st.warning(answer)
    else:
        st.success(answer)

    # SECTION 3 — Source Documents
    st.subheader("Source Documents")
    if result["results"]:
        for i, r in enumerate(result["results"], 1):
            m = r["metadata"]
            with st.expander(
                f"[{i}] {m['file_name']} — {m['author']} ({m['date']}) | distance: {r['distance']:.3f}"
            ):
                st.markdown(f"**File:** {m['file_name']}")
                st.markdown(f"**Author:** {m['author']}")
                st.markdown(f"**Date:** {m['date']}")
                st.markdown(f"**Type:** {m['document_type']}")
                st.markdown(f"**Distance:** {r['distance']:.4f}")
                st.markdown("**Content:**")
                st.text(r["content"])
    else:
        st.info("No source documents retrieved.")

    # SECTION 4 — Debug Info
    if st.checkbox("Show Debug Info", value=False):
        st.markdown("---")
        st.subheader("Debug Info")
        st.markdown(f"**Chunks retrieved:** {len(result['results'])}")
        st.markdown("**Raw Intent:**")
        st.json(result["intent"])
        st.markdown("**Raw Filter:**")
        st.json(result["filter"] if result["filter"] else {})

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
