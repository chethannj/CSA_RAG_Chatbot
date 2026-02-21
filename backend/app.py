from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import build_chain
from ingest import ingest

app = FastAPI(title="Customer Support Agent RAG Chatbot")

rag = None  # Don't build at startup

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: QueryRequest):
    global rag
    try:
        if rag is None:
            rag = build_chain()
        return rag(request.question)
    except ValueError as e:
        return {"answer": str(e), "sources": [], "error": "ChromaDB empty - run /ingest first"}
    except Exception as e:
        return {"answer": "An error occurred.", "sources": [], "error": str(e)}

@app.post("/ingest")
def run_ingestion():
    global rag
    try:
        ingest()
        rag = None  # Reset so it rebuilds after ingestion
        return {"status": "success", "message": "Ingestion completed successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}