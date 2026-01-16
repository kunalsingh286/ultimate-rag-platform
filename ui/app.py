import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Ultimate RAG Platform", layout="wide")

st.title("🚀 Ultimate RAG Platform")
st.caption("Production-grade Retrieval-Augmented Generation using open-source LLMs")

# -------------------------------
# Sidebar: Document Upload
# -------------------------------
st.sidebar.header("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a document (PDF / TXT / MD)",
    type=["pdf", "txt", "md"],
)

if uploaded_file:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    with st.sidebar.spinner("Uploading and indexing document..."):
        resp = requests.post(f"{API_BASE}/upload", files=files)

    if resp.status_code == 200:
        st.sidebar.success("Document indexed successfully!")
        st.sidebar.json(resp.json())
    else:
        st.sidebar.error("Failed to upload document")

# -------------------------------
# Main: Query Interface
# -------------------------------
st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="e.g. What is Retrieval-Augmented Generation?",
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            resp = requests.post(
                f"{API_BASE}/query",
                params={"question": question},
            )

        if resp.status_code != 200:
            st.error("Error querying RAG backend")
        else:
            data = resp.json()

            st.markdown("### ✅ Answer")
            st.write(data.get("answer", ""))

            st.markdown("### 📊 Confidence")
            st.progress(min(max(float(data.get("confidence", 0.0)), 0.0), 1.0))

            st.markdown("### 📚 Sources")
            for i, src in enumerate(data.get("sources", []), 1):
                with st.expander(f"Source {i}"):
                    st.write(src.get("content", ""))

            if not data.get("found", True):
                st.warning("Answer not found in provided documents.")
