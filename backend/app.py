from fastapi import FastAPI

app = FastAPI(title="Customer Support Agent RAG Chatbot")

@app.get("/health")
def health_check():
    return {"status": "ok"}

