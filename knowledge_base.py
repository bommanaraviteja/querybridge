import uuid
import requests
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from bs4 import BeautifulSoup

# --- Confluence configuration ---
CONFLUENCE_USER_NAME = ""
CONFLUENCE_URL = ""
CONFLUENCE_TOKEN= ""


def run(llm, embedding_model, vectordb):
    st.title("📚 Knowledge Base")
    st.write("Welcome to Knowledge Base")
    st.write("Upload and manage internal documents. This is where your content gets indexed and prepared for querying.")

    # --- Source selection ---
    option = st.selectbox("Select Source", ["Confluence", "JIRA", "Standalone PDF"])

    # -----------------------
    # Standalone PDF handling
    # -----------------------
    if option == "Standalone PDF":
        uploaded_files = st.file_uploader("Upload your Knowledge Base", accept_multiple_files=True, type=["pdf"])
        if uploaded_files:
            with st.spinner("Processing PDF files... 🚀"):
                for file in uploaded_files:
                    process_pdf(file, embedding_model, vectordb)
            st.success("✅ PDF documents successfully added to vector DB.")

    # -----------------------
    # JIRA placeholder
    # -----------------------
    elif option == "JIRA":
        st.info("🚧 JIRA integration is work in progress. Stay tuned!")

    # -----------------------
    # Confluence Integration
    # -----------------------
    elif option == "Confluence":
        # Initialize session state variables
        if "spaces" not in st.session_state:
            st.session_state.spaces = None
        if "selected_space" not in st.session_state:
            st.session_state.selected_space = None
        if "spaces_loaded" not in st.session_state:
            st.session_state.spaces_loaded = False

        # Allow user to override default Confluence credentials
        url = st.text_input("Confluence Base URL", CONFLUENCE_URL)
        username = st.text_input("Confluence Username/Email", CONFLUENCE_USER_NAME)
        token = st.text_input("Confluence API Token", type="password", value=CONFLUENCE_TOKEN)

        # Fetch spaces on button click
        if st.button("Fetch Spaces"):
            if not url or not username or not token:
                st.warning("Please provide all Confluence credentials.")
            else:
                spaces = fetch_confluence_spaces(url, username, token)
                if spaces:
                    st.session_state.spaces = spaces
                    st.session_state.spaces_loaded = True
                    st.success("✅ Spaces fetched successfully!")

        # Show spaces dropdown if loaded
        if st.session_state.spaces_loaded and st.session_state.spaces:
            st.session_state.selected_space = st.selectbox(
                "Select a Space",
                list(st.session_state.spaces.keys()),
                key="space_dropdown"
            )

            if st.button("Load Pages from Selected Space"):
                selected_space = st.session_state.selected_space
                with st.spinner(f"Loading and parsing pages from '{selected_space}'..."):
                    process_confluence_space(
                        url,
                        username,
                        token,
                        st.session_state.spaces[selected_space],
                        embedding_model,
                        vectordb
                    )
                st.success(f"✅ All pages and PDF attachments from '{selected_space}' have been processed.")


# -----------------------
# Helper Functions
# -----------------------

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
    if not chunks:
        return

    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embedding_model.encode(chunks).tolist()
    vectordb.add(documents=chunks, embeddings=embeddings, ids=ids)


def fetch_confluence_spaces(base_url, username, token):
    url = f"{base_url}/rest/api/space"
    response = requests.get(url, auth=(username, token))
    if response.status_code == 200:
        data = response.json()
        return {space['name']: space['key'] for space in data.get('results', [])}
    else:
        st.error(f"Failed to fetch spaces: {response.status_code} - {response.text}")
        return None


def process_confluence_space(base_url, username, token, space_key, embedding_model, vectordb):
    page_url = f"{base_url}/rest/api/content"
    params = {
        "spaceKey": space_key,
        "expand": "body.storage,version,metadata.labels,children.attachment",
        "limit": 100
    }

    response = requests.get(page_url, params=params, auth=(username, token))
    if response.status_code != 200:
        st.error(f"Failed to fetch pages: {response.status_code} - {response.text}")
        return

    pages = response.json().get("results", [])
    for page in pages:
        title = page.get("title", "Untitled")
        st.write(f"📄 Processing page: {title}")
        content_html = page.get("body", {}).get("storage", {}).get("value", "")
        plain_text = extract_plain_text(content_html)
        if plain_text.strip():
            process_text(plain_text, embedding_model, vectordb)
        process_text(content_html, embedding_model, vectordb)

        # Process PDF attachments
        attachments = page.get("children", {}).get("attachment", {}).get("results", [])
        for attachment in attachments:
            if attachment["metadata"]["mediaType"] == "application/pdf":
                st.info("📎 PDF attachments found. Downloading and parsing attachments...")
                download_link = f"{base_url}{attachment['_links']['download']}"
                pdf_response = requests.get(download_link, auth=(username, token))
                if pdf_response.status_code == 200:
                    pdf_file = BytesIO(pdf_response.content)
                    process_pdf(pdf_file, embedding_model, vectordb)


def process_text(text, embedding_model, vectordb):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=5000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        return

    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embedding_model.encode(chunks).tolist()
    vectordb.add(documents=chunks, embeddings=embeddings, ids=ids)

def extract_plain_text(html_content):
    """Remove HTML tags and keep readable plain text."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script/style tags if any
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator="\n")  # Preserve line breaks
    # Collapse excessive whitespace
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())