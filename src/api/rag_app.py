# src/api/rag_app.py
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline  # no need for RAGResult, RAGSource

app = FastAPI(title="SupplementsRx RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


# ✅ Response model: ONLY the chatbot's answer
class AnswerResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    # Very simple check; you can add more if you want
    return {"status": "ok"}


@app.post("/api/query", response_model=AnswerResponse)
async def query_agent(payload: QueryRequest):
    try:
        # use the helper that already returns {"answer": "..."}
        result_dict = pipeline.run_as_dict(
            question=payload.question,
            top_k=payload.top_k,
        )
        # result_dict = {"answer": "..."}
        return AnswerResponse(**result_dict)

    except Exception as e:
        # For debugging in PowerShell / frontend
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": e.__class__.__name__,
            },
        )
