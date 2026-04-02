from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import build_index, query_pipeline

app = FastAPI(title="Venio Smart API", version="1.0.0")
store = None


@app.on_event("startup")
async def startup_event():
    global store
    store = build_index()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Venio Smart"}


@app.post("/query")
def query_documents(request: QueryRequest):
    try:
        result = query_pipeline(request.query, store, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
