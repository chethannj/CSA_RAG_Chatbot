from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import build_chain
from ingest import ingest

app = FastAPI(title="Customer Support Agent RAG Chatbot")

rag=build_chain()

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: QueryRequest):
    try:
        return rag(request.question)
    except Exception as e:
        return {
            "answer": "An error occurred while processing your request.",
            "sources": [],
            "error": str(e)
        }
@app.post("/ingest")
def run_ingestion():
     """
    Ingests all docs from Data/ into ChromaDB.
    """
     try:
        ingest()
        return {"status": "success", "message": "Ingestion completed successfully."}
     except Exception as e:
        return {"status": "error", "message": str(e)}
   

