# src/api/unified_rag_app.py

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from datetime import datetime

from src.rag.unified_pipeline import UnifiedRAGPipeline

app = FastAPI(title="SupplementsRx Unified RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = UnifiedRAGPipeline()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class RatingRequest(BaseModel):
    question: str
    answer: str
    rating: int  # 1 for thumbs up, -1 for thumbs down
    feedback: Optional[str] = None

class AccuracyReportRequest(BaseModel):
    question: str
    answer: str
    reason: Optional[str] = None  # Optional reason for why it's inaccurate

class AnswerResponse(BaseModel):
    answer: str
    reasoning: str
    sources: List[Dict[str, Any]]
    used_llm_knowledge: bool
    precaution_notice: str
    natmed_sections: Optional[List[str]] = None

class RatingResponse(BaseModel):
    status: str
    message: str

class AccuracyReportResponse(BaseModel):
    status: str
    message: str


# Simple in-memory storage for ratings (in production, use a database)
ratings_store = []
accuracy_reports_store = []

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Test Neo4j connection
        if pipeline.neo4j_driver:
            with pipeline.neo4j_driver.session() as session:
                session.run("RETURN 1")
        return {
            "status": "ok", 
            "llm": pipeline.llm or "fallback",
            "llm_model": pipeline.llm_model if hasattr(pipeline, 'llm_model') else None
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/api/query", response_model=AnswerResponse)
async def query_agent(payload: QueryRequest):
    """
    Main query endpoint that combines vector embeddings and Neo4j KG
    """
    try:
        result = pipeline.run(payload.question)
        
        # Extract NatMed sections from sources
        natmed_sections = []
        for source in result.sources:
            if source.section:
                natmed_sections.append(source.section)
        natmed_sections = list(set(natmed_sections))  # Remove duplicates
        
        return AnswerResponse(
            answer=result.answer,
            reasoning=result.reasoning,
            sources=[{
                "source_type": s.source_type,
                "supplement_name": s.supplement_name,
                "section": s.section,
                "text": s.text[:200] if s.text else "",  # Truncate for response
                "source_url": s.source_url,
                "score": s.score,
            } for s in result.sources],
            used_llm_knowledge=result.used_llm_knowledge,
            precaution_notice=result.precaution_notice,
            natmed_sections=natmed_sections,
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": e.__class__.__name__,
            },
        )

@app.post("/api/rate", response_model=RatingResponse)
async def rate_response(payload: RatingRequest):
    """
    Rate a response (thumbs up/down)
    """
    try:
        if payload.rating not in [1, -1]:
            raise HTTPException(status_code=400, detail="Rating must be 1 (thumbs up) or -1 (thumbs down)")
        
        rating_entry = {
            "question": payload.question,
            "answer": payload.answer[:500],  # Truncate for storage
            "rating": payload.rating,
            "feedback": payload.feedback,
            "timestamp": datetime.now().isoformat(),
        }
        
        ratings_store.append(rating_entry)
        
        # In production, save to database
        # For now, just log it
        print(f"Rating received: {rating_entry}")
        
        return RatingResponse(
            status="success",
            message="Thank you for your feedback!"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": e.__class__.__name__,
            },
        )

@app.get("/api/ratings")
async def get_ratings():
    """
    Get all ratings (for admin/debugging purposes)
    """
    return {
        "total": len(ratings_store),
        "ratings": ratings_store[-100:],  # Last 100 ratings
    }

@app.post("/api/report-accuracy", response_model=AccuracyReportResponse)
async def report_accuracy(payload: AccuracyReportRequest):
    """
    Report an inaccurate answer
    """
    try:
        accuracy_report = {
            "question": payload.question,
            "answer": payload.answer[:500],  # Truncate for storage
            "reason": payload.reason,
            "timestamp": datetime.now().isoformat(),
        }
        
        accuracy_reports_store.append(accuracy_report)
        
        # In production, save to database
        # For now, just log it
        print(f"Accuracy report received: {accuracy_report}")
        
        return AccuracyReportResponse(
            status="success",
            message="Thank you for reporting this. We'll review it to improve our answers."
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": e.__class__.__name__,
            },
        )

@app.get("/api/accuracy-reports")
async def get_accuracy_reports():
    """
    Get all accuracy reports (for admin/debugging purposes)
    """
    return {
        "total": len(accuracy_reports_store),
        "reports": accuracy_reports_store[-100:],  # Last 100 reports
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    pipeline.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
