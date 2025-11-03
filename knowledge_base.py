import uuid

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
CONFLUENCE_TOKEN=""
def run(llm, embedding_model, vectordb):
    st.title("📚 Knowledge Base")
    st.write("Welcome to Knowledge Base")
    st.write("Upload and manage internal documents. This is where your content gets indexed and prepared for querying.")
    option = st.selectbox("Select Source", ["Confluence", "JIRA", "Standalone PDF"])
    if option == "Standalone PDF":
        uploaded_files = st.file_uploader("Upload your Knowledge Base", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            with st.spinner("Processing PDF files... 🚀"):
                for file in uploaded_files:
                    process_pdf(file, embedding_model, vectordb)
            st.success("✅ PDF documents successfully added to vector DB.")



def process_pdf(file, embedding_model, vectordb):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embedding_model.encode(chunks).tolist()
    vectordb.add(documents=chunks, embeddings=embeddings, ids=ids)
