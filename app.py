import streamlit as st
from importlib import import_module

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


@st.cache_resource
def init_resources():
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = PersistentClient(path="./chroma_storage")
    collection = client.get_or_create_collection("my_docs")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key="",
        temperature=0.2
    )
    return llm, emb_model, collection

llm, embedding_model, vectordb = init_resources()

# Page title
st.set_page_config(page_title="Query Bridge", layout="wide")

NAV_ROUTES = {
    "Home": "home",
    "Playground": "playground",
    "Knowledge Base": "knowledge_base"
}

# Left navigation
selection = st.sidebar.selectbox("Go to", NAV_ROUTES.keys())
page_module = import_module(NAV_ROUTES[selection])
page_module.run(llm, embedding_model, vectordb)

