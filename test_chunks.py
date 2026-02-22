from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

db = Chroma(
    persist_directory=r"D:\RAG\TCS_RAG\chroma_db",
    embedding_function=OllamaEmbeddings(model="mxbai-embed-large:latest")
)

# Total count
print(f"Total chunks: {db._collection.count()}")

# Preview first 3 chunks
docs = db.get(include=["documents", "metadatas"], limit=10)
for i, (text, meta) in enumerate(zip(docs["documents"], docs["metadatas"])):
    print(f"\n--- Chunk {i+1} | Page: {meta['page']} | Source: {meta['source']} ---")
    print(text[:400])  # first 400 chars
    print("...")