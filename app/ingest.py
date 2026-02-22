"""Index TCS financial PDFs into ChromaDB"""

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Config
EMBEDDING_MODEL = "mxbai-embed-large:latest"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (TCS_RAG/)
PDF_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def ingest_pdfs():
    # Ensure data directory exists
    if not os.path.exists(PDF_DIR):
        print(f"❌ Error: Create a '{PDF_DIR}' folder and put your TCS PDFs there.")
        return

    all_filtered_chunks = []

# 1. Page Selection Logic (Based on your Index)
    # We target: Performance (8-16), Statutory (39-127), and Consolidated (319-323)
    # Note: Python slices are 0-indexed, so Page 8 is index 7.
    target_pages = (
        list(range(7, 16)) +    # Performance Review
        list(range(38, 127)) +  # Statutory Section
        list(range(318, 323))   # Consolidated Financials
    )
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith('.pdf'):
            file_path = os.path.join(PDF_DIR, pdf_file)
            print(f"📂 Loading: {pdf_file}")
            
            # Using PyMuPDF for speed and better metadata
            loader = PyMuPDFLoader(file_path)
            full_doc = loader.load()
            
            # Filter only the important pages
            important_docs = [full_doc[i] for i in target_pages if i < len(full_doc)]

            # 2. Financial-Optimized Chunking
            # We use a larger chunk size to keep financial tables together
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,    # Large enough for tables
                chunk_overlap=200,   # Keep context between chunks
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = splitter.split_documents(important_docs)

            # 3. Enhanced Metadata
            for chunk in chunks:
                chunk.metadata.update({
                    "source": pdf_file,
                    "page": chunk.metadata.get("page", 0) + 1, # Human-readable page #
                    "category": "Financial/Statutory"
                })
            
            all_filtered_chunks.extend(chunks)

    if not all_filtered_chunks:
        print("⚠️ No relevant chunks found. Check your PDF page ranges.")
        return
    
    # 4. Vector Store Creation (Local Embeddings)
    print(f"🧠 Generating embeddings using {EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Chroma v0.6+ handles persistence automatically
    vectorstore = Chroma.from_documents(
        documents=all_filtered_chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print(f"✅ Success! Indexed {len(all_filtered_chunks)} chunks into {DB_PATH}")

if __name__ == "__main__":
    ingest_pdfs()
