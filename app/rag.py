from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL = "mxbai-embed-large:latest"
CHAT_MODEL = "llama3.1:8b"
DB_PATH = "../chroma_db"

class TCSRAG:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.llm = ChatOllama(model=CHAT_MODEL, temperature=0.1)
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a TCS financial analyst. Answer using ONLY provided data.
        
        CONTEXT: {context}
        QUESTION: {question}
        
        Format: **Answer**: [exact value] [period] [Source: page X]
        """)
        
        self.rag_chain = (
            {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
             "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        return self.rag_chain.invoke(question)