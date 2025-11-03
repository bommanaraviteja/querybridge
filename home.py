import streamlit as st

def run(*args, **kwargs):
    st.title("📄 Query Bridge")
    st.subheader("Bridge the Gap Between Questions and Knowledge.")

    st.markdown("""
        **Query Bridge** is an AI-powered tool that lets you query your internal documents using natural language.  
        It uses a Retrieval-Augmented Generation (RAG) approach to provide accurate, context-aware answers from your own content.
        """)

    st.markdown("---")

    st.markdown("### 🧭 How It Works")

    st.markdown("""
        - 📁 **Knowledge Base** – Upload and manage internal documents. This is where your content gets indexed and prepared for querying.  
        - 🎮 **Playground** – Ask questions using plain English. Get AI-powered answers, grounded in the documents you've uploaded.
        """)

    st.markdown("---")

    st.markdown("### 🚀 Why Use Query Bridge?")
    st.markdown("""
        - 🔍 Natural language search across all your documents  
        - 🤖 AI-generated answers with full context and citations  
        - 🔐 Secure and private — your data stays with you  
        - 💡 Great for teams, support, research, and internal tools
        """)

    st.markdown("---")

    st.markdown(
        "<center><strong>Start by uploading documents in the Knowledge Base. Then ask away in the Playground.</strong></center>",
        unsafe_allow_html=True)