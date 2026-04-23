import asyncio
from fastapi import APIRouter, HTTPException
from backend.app.schemas import QueryRequest, QueryResponse, RetrievedTicket
from backend.app.services.vector_store import VectorStoreService
from backend.app.services.llm_service import LLMService
from backend.app.services.ml_inference import MLInferenceService
from backend.app.config import settings

router = APIRouter()

# Initialize services (singleton instances)
vector_store = VectorStoreService()
llm = LLMService()
ml = MLInferenceService()

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    query = request.text
    
    # 1. Retrieve similar tickets
    retrieved = vector_store.query(query)
    sources = []
    for doc, meta, dist in zip(retrieved["documents"], retrieved["metadatas"], retrieved["distances"]):
        # Cosine distance (0=identical, 2=opposite)
        similarity = 1 - (dist / 2)  # convert to 0-1 where 1 is most similar
        if similarity >= settings.SIMILARITY_THRESHOLD:
            sources.append(RetrievedTicket(
                text=doc,
                priority=meta.get("priority", "unknown"),
                similarity=round(similarity, 4)
            ))
    
    # 2. Build prompts
    rag_prompt = _build_rag_prompt(query, sources)
    non_rag_prompt = f"Answer this customer support question concisely:\n\n{query}"
    llm_priority_prompt = f"""Is the following support ticket urgent or normal? Reply with only one word: "urgent" or "normal". Then provide a brief reason.

Ticket: {query}
"""
    
    # 3. Run all four tasks in parallel
    rag_task = asyncio.to_thread(llm.generate, rag_prompt, "You are a helpful customer support assistant. Use the provided past tickets as context.")
    non_rag_task = asyncio.to_thread(llm.generate, non_rag_prompt, "You are a helpful customer support assistant.")
    ml_task = asyncio.to_thread(ml.predict, query)
    llm_priority_task = asyncio.to_thread(llm.generate, llm_priority_prompt, "You are a ticket triage system.")
    
    rag_res, non_rag_res, ml_res, llm_prio_res = await asyncio.gather(
        rag_task, non_rag_task, ml_task, llm_priority_task
    )
    
    # 4. Parse LLM priority response
    llm_pred_text = llm_prio_res["content"].strip().lower()
    if "urgent" in llm_pred_text:
        pred = "urgent"
    else:
        pred = "normal"
    reasoning = llm_pred_text.split('\n', 1)[-1] if '\n' in llm_pred_text else llm_pred_text
    
    return QueryResponse(
        query=query,
        rag_answer={
            "answer": rag_res["content"],
            "sources": sources,
            "latency_ms": rag_res["latency_ms"],
            "cost_usd": rag_res["cost_usd"]
        },
        non_rag_answer={
            "answer": non_rag_res["content"],
            "latency_ms": non_rag_res["latency_ms"],
            "cost_usd": non_rag_res["cost_usd"]
        },
        ml_priority=ml_res,
        llm_priority={
            "prediction": pred,
            "reasoning": reasoning,
            "latency_ms": llm_prio_res["latency_ms"],
            "cost_usd": llm_prio_res["cost_usd"]
        }
    )

def _build_rag_prompt(query: str, sources: list[RetrievedTicket]) -> str:
    context = "\n\n".join([f"Past Ticket: {s.text}" for s in sources[:3]])
    return f"""Use the following past support tickets to help answer the new question.
    
Past Tickets:
{context}

New Question: {query}

Answer:"""