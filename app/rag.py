import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Config — loaded from .env (single source of truth for model names)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root (TCS_RAG/)
load_dotenv(os.path.join(BASE_DIR, ".env"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

class TCSRAG:
    def __init__(self):
        # Check that vector store exists and has data
        if not os.path.exists(DB_PATH) or not os.listdir(DB_PATH):
            raise RuntimeError(
                f"❌ Vector store not found at {DB_PATH}.\n"
                f"   Run 'python app/ingest.py' first to index your PDFs."
            )

        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # Fail fast if Ollama is not reachable
        try:
            self.embeddings.embed_query("test")
        except Exception as e:
            raise RuntimeError(
                f"❌ Cannot connect to Ollama: {e}\n"
                f"   → ollama serve\n"
                f"   → ollama pull {EMBEDDING_MODEL}\n"
                f"   → ollama pull {CHAT_MODEL}"
            )

        self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a TCS financial analyst. Answer using ONLY the provided context data.
        
        Rules:
        - If the answer is NOT in the context, say "Cannot determine from available data"
        - Do NOT guess, estimate, or infer beyond what the data shows
        - Include exact values with their time period (e.g., FY2025)
        - Cite source page numbers for every factual claim
        - For calculations, show your work using only retrieved data
        
        CONTEXT: {context}
        QUESTION: {question}
        
        Format: **Answer**: [exact value] [period] [Source: page X]
        """)
        
        self.rag_chain = (
            {"context": self.retriever | (lambda docs: "\n\n".join(
                f"[Page {d.metadata.get('page', '?')} | {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
            )), 
             "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        try:
            return self.rag_chain.invoke(question)
        except Exception as e:
            return f"❌ Query failed: {e}"