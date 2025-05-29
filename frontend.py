import streamlit as st
import requests
import time
from typing import List, Dict, Optional
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Document Research Chatbot", 
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/Ayushi-bhutani/Chatbot_theme_Identifier',
        'Report a bug': "mailto:divyansh.sharma@thewasserstoff.com",
        'About': "# Wasserstoff AI Intern Project"
    }
)

API_BASE = "http://localhost:8000"  # FastAPI base URL

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton>button { background-color: #4CAF50; color: white; }
    .stAlert { padding: 20px; }
    .stProgress > div > div > div > div { background-color: #4CAF50; }
    .stTable { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("📄 Document Research & Theme Identification Chatbot")
st.markdown("""
Welcome! Upload your documents (PDFs or scanned images) and ask questions to get  
detailed, cited answers synthesized across your document collection.
""")

# --- 1. Upload Documents ---
st.header("1. Upload Documents (PDF / Images)")

with st.expander("Upload Options", expanded=True):
    uploaded_files = st.file_uploader(
        "Select multiple PDF or image files (minimum 75 for full functionality):",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload at least 75 documents for comprehensive theme analysis"
    )

    if uploaded_files:
        if st.button("Upload & Process Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for i, file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
                    files = {"file": (file.name, file.getvalue(), file.type)}
                    resp = requests.post(f"{API_BASE}/upload/", files=files)

                    if resp.status_code == 200:
                        results.append({
                            "filename": file.name,
                            "status": "success",
                            "message": resp.json().get("message", "Uploaded successfully")
                        })
                    else:
                        results.append({
                            "filename": file.name,
                            "status": "failed",
                            "message": resp.text
                        })

                except Exception as e:
                    results.append({
                        "filename": file.name,
                        "status": "failed",
                        "message": str(e)
                    })

                progress_bar.progress((i + 1) / len(uploaded_files))
                time.sleep(0.1)

            success_count = sum(1 for r in results if r["status"] == "success")
            failed_results = [r for r in results if r["status"] == "failed"]

            if success_count > 0:
                st.success(f"✅ Successfully uploaded {success_count}/{len(uploaded_files)} files")

            if failed_results:
                st.subheader("❌ Upload Errors")
                for result in failed_results:
                    st.error(f"{result['filename']}: {result['message']}")

# --- 2. Uploaded Documents Overview ---
st.header("2. Uploaded Documents Overview")

@st.cache_data(ttl=60)
def get_documents():
    try:
        resp = requests.get(f"{API_BASE}/documents/", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("documents", [])
        return []
    except Exception:
        return []

docs = get_documents()

if docs:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", len(docs))
    col2.metric("Total Pages", sum(d.get("page_count", 0) for d in docs))
    col3.metric("Last Upload", docs[0].get("upload_date", "N/A"))

    search_term = st.text_input("Search documents:")
    filtered_docs = [d for d in docs if search_term.lower() in d["filename"].lower()] if search_term else docs

    st.dataframe(
        [
            {
                "Filename": doc["filename"],
                "Pages": doc["page_count"],
                "Uploaded": doc["upload_date"],
                "Size": f"{doc.get('size_kb', 0)} KB",
                "Type": "Scanned" if doc.get("is_scanned") else "Digital"
            }
            for doc in filtered_docs
        ],
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No documents uploaded yet. Upload documents in section 1.")

# --- 3. Ask a Question ---
st.header("3. Ask a Question")

query = st.text_area(
    "Enter your question about the uploaded documents:",
    placeholder="E.g., What are the key findings about liver disease treatment?",
    height=100
)

with st.expander("🔍 Advanced Query Options"):
    col1, col2 = st.columns(2)
    with col1:
        semantic_search = st.checkbox("Semantic Search", value=True)
        keyword_search = st.checkbox("Keyword Search", value=False)
        limit_results = st.slider("Results per document", 1, 20, 5)
    with col2:
        selected_docs = []
        if docs:
            doc_options = [d["filename"] for d in docs]
            selected_docs = st.multiselect("Target specific documents (optional):", doc_options)

if st.button("Search Documents", type="primary", disabled=not (query and docs)):
    with st.spinner("Analyzing documents..."):
        try:
            params = {
                "question": query,
                "semantic": str(semantic_search).lower(),
                "keyword": str(keyword_search).lower(),
                "limit": limit_results
            }
            if selected_docs:
                params["documents"] = ",".join(selected_docs)

            start = time.time()
            response = requests.get(f"{API_BASE}/query/", params=params, timeout=30)
            duration = time.time() - start

            if response.status_code == 200:
                data = response.json()

                st.subheader("🔎 Search Results")
                st.caption(f"Found {len(data.get('results', []))} sections in {duration:.2f} sec")

                with st.expander("📑 Detailed Results", expanded=True):
                    for result in data.get("results", []):
                        st.markdown(f"**📄 {result['document']}**, Page {result['page']}")
                        st.markdown(f"*Relevance: {result['score']:.2f}*")
                        st.markdown(f"> {result['excerpt']}")
                        st.code(result['citation'])
                        st.divider()

                if data.get("themes"):
                    st.subheader("🧠 Synthesized Themes")
                    for theme in data["themes"]:
                        with st.expander(f"{theme['theme']} ({len(theme['documents'])} docs)"):
                            st.markdown(theme["gpt_summary"])
                            st.caption("Supported by: " + ", ".join(theme["documents"]))
            else:
                st.error(f"❌ Search failed: {response.text}")

        except requests.exceptions.Timeout:
            st.error("⏳ Query timed out.")
        except Exception as e:
            st.error(f"🚫 Error: {str(e)}")

# --- 4. Theme Network Visualization ---
st.header("4. Document Theme Network Visualization")


st.components.v1.html(
    f"""
    <iframe src="{API_BASE}/api/visualize" width="100%" height="700px" style="border:none;"></iframe>
    """, height=720
)


# --- Footer ---
st.markdown("---")
st.markdown("""
<small style="color:gray">
    Powered by [Wasserstoff](https://www.thewasserstoff.com/) — AI Intern Project  
    Need help? Contact: divyansh.sharma@thewasserstoff.com
</small>
""", unsafe_allow_html=True)
