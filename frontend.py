import streamlit as st
import requests

st.title("📄 Document Research Chatbot")
st.subheader("Upload a PDF and ask questions!")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file is not None:
    with st.spinner("Uploading and processing..."):
        response = requests.post(
            "http://localhost:8000/upload/",
            files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        )
        if response.status_code == 200:
            data = response.json()
            st.success(f"{data['message']}")
            st.json(data["metadata"])
        else:
            st.error(f"Upload failed: {response.text}")

# --- Query Section ---
query = st.text_input("🔍 Ask a question about the uploaded documents:")
if query:
    with st.spinner("Searching..."):
        response = requests.post(
            "http://localhost:8000/query/",
            params={
                "question": query,
                "semantic": True,
                "keyword": False,
                "limit": 5
            }
        )
        if response.status_code == 200:
            results = response.json()["results"]
            if results:
                for res in results:
                    st.markdown(f"**📄 {res['document']} (Page {res['page']})**")
                    st.write(res["excerpt"])
                    st.caption(f"Relevance Score: {res['score']:.2f}")
            else:
                st.warning("No relevant results found.")
        else:
            st.error(f"Search failed: {response.text}")
