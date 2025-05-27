import streamlit as st
import requests
from typing import List, Dict

# Set page config
st.set_page_config(page_title="Document Research Chatbot", layout="wide")

API_BASE = "http://localhost:8000"  # Your backend API URL

st.title("📄 Document Research & Theme Identification Chatbot")
st.markdown("""
Welcome! Upload your documents (PDFs or scanned images) and ask questions to get  
detailed, cited answers synthesized across your document collection.
""")

# --- 1. Upload documents ---
st.header("1. Upload Documents (PDF / Images)")

uploaded_files = st.file_uploader(
    "Select multiple PDF or image files (minimum 75 for full functionality):",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.button("Upload & Process Documents"):
        with st.spinner(f"Uploading {len(uploaded_files)} files..."):
            success_count = 0
            fail_count = 0
            for file in uploaded_files:
                files = {"file": (file.name, file, "application/octet-stream")}
                try:
                    resp = requests.post(f"{API_BASE}/upload/", files=files)
                    if resp.status_code == 200:
                        success_count += 1
                    else:
                        fail_count += 1
                except Exception:
                    fail_count += 1
            st.success(f"Successfully uploaded {success_count} files.")
            if fail_count > 0:
                st.error(f"Failed to upload {fail_count} files. Please try again.")

# --- 2. Document management ---

st.header("2. Uploaded Documents Overview")

try:
    resp = requests.get(f"{API_BASE}/documents/")
    if resp.status_code == 200:
        docs = resp.json().get("documents", [])
        if docs:
            # Show documents table with metadata
            st.table([
                {
                    "Document ID": doc.get("id", "N/A"),
                    "Filename": doc.get("filename", "N/A"),
                    "Upload Date": doc.get("upload_date", "N/A"),
                    "Pages": doc.get("pages", "N/A"),
                    "Author": doc.get("author", "Unknown"),
                    "Type": doc.get("type", "Unknown"),
                }
                for doc in docs
            ])
        else:
            st.info("No documents uploaded yet.")
    else:
        st.error("Failed to load documents metadata.")
except Exception as e:
    st.error(f"Error fetching documents: {str(e)}")

# --- 3. Query input & options ---

st.header("3. Ask a Question")

query = st.text_input("Enter your question about the uploaded documents:")

with st.expander("Advanced Query Options"):
    semantic_search = st.checkbox("Enable Semantic Search", value=True)
    keyword_search = st.checkbox("Enable Keyword Search", value=False)
    # Document selection for targeted querying (fetched dynamically)
    selected_docs = []
    if docs:
        doc_options = [f"{doc['id']} - {doc['filename']}" for doc in docs]
        selected_docs = st.multiselect("Select documents to query (leave empty for all):", options=doc_options)

limit_results = st.slider("Max Results Per Document", min_value=1, max_value=20, value=5)

if st.button("Search"):
    if not query:
        st.warning("Please enter a question to proceed.")
    else:
        with st.spinner("Searching documents and synthesizing response..."):
            params = {
                "question": query,
                "semantic": semantic_search,
                "keyword": keyword_search,
                "limit": limit_results,
            }
            if selected_docs:
                # Extract doc IDs from selection format "id - filename"
                doc_ids = [doc.split(" - ")[0] for doc in selected_docs]
                params["documents"] = ",".join(doc_ids)

            try:
                response = requests.post(f"{API_BASE}/query/", params=params)
                if response.status_code == 200:
                    data = response.json()

                    # Show detailed results per document
                    st.subheader("Per-Document Results with Citations")
                    for doc_result in data.get("results_by_document", []):
                        st.markdown(f"### Document {doc_result['document_id']} - {doc_result['document_name']}")
                        for excerpt in doc_result.get("excerpts", []):
                            st.markdown(f"- Page {excerpt['page']}, Para {excerpt['paragraph']}")
                            st.write(excerpt['text'])
                            st.caption(f"Relevance Score: {excerpt['score']:.2f}")

                    # Show final synthesized themes & answers
                    st.subheader("Synthesized Themes & Final Answer")
                    for theme in data.get("themes", []):
                        st.markdown(f"**Theme:** {theme['theme_name']}")
                        st.write(theme["synthesized_text"])
                        st.markdown("**Citations:**")
                        for cite in theme.get("citations", []):
                            st.markdown(f"- Document {cite['document_id']}, Page {cite['page']}, Para {cite['paragraph']}")

                else:
                    st.error(f"Query failed with error: {response.text}")

            except Exception as e:
                st.error(f"Error during query processing: {str(e)}")

st.markdown("---")
st.caption("Powered by Wasserstoff - AI Intern Project")
