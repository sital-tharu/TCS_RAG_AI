"""Index TCS financial PDFs into ChromaDB"""

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Config
EMBEDDING_MODEL = "mxbai-embed-large:latest"
PDF_DIR = "data"
DB_PATH = "./chroma_db"

def ingest_pdfs():
    # Ensure data directory exists
    if not os.path.exists(PDF_DIR):
        print(f"❌ Error: Create a '{PDF_DIR}' folder and put your TCS PDFs there.")
        return

    all_filtered_chunks = []