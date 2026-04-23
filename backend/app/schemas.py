from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    text: str

class RetrievedTicket(BaseModel):
    text: str
    priority: str
    similarity: float  # distance, lower is better for cosine

class RAGAnswer(BaseModel):
    answer: str
    sources: List[RetrievedTicket]
    latency_ms: float
    cost_usd: float

class NonRAGAnswer(BaseModel):
    answer: str
    latency_ms: float
    cost_usd: float

class MLPrediction(BaseModel):
    prediction: str
    confidence: float
    latency_ms: float
    cost_usd: float

class LLMPrediction(BaseModel):
    prediction: str
    reasoning: str
    latency_ms: float
    cost_usd: float

class QueryResponse(BaseModel):
    query: str
    rag_answer: RAGAnswer
    non_rag_answer: NonRAGAnswer
    ml_priority: MLPrediction
    llm_priority: LLMPrediction