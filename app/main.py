import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Load config from .env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:8b")

try:
    from .rag import TCSRAG       # works with: python -m app.main
except ImportError:
    from rag import TCSRAG        # works with: python main.py

app = FastAPI(title="TCS Financial RAG API", version="1.0.0")

# Global RAG instance (load once)
rag = TCSRAG()

class QueryRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 500

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": CHAT_MODEL}

@app.post("/query")
async def query_tcs(request: QueryRequest):
    try:
        response = rag.query(request.question)
        return {
            "question": request.question,
            "answer": response,
            "tokens": request.max_tokens
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs")
async def docs_redirect():
    return {"message": "Visit /docs for Swagger UI"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
