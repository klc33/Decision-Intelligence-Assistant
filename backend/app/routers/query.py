import asyncio
from fastapi import APIRouter
from app.schemas import QueryRequest, QueryResponse, RetrievedTicket
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.services.ml_inference import MLInferenceService
from app.config import settings

router = APIRouter()

vector_store = VectorStoreService()
llm = LLMService()
ml = MLInferenceService()

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    query = request.text
    
    # 1. Retrieve
    retrieved = vector_store.query(query)
    sources = []
    for doc, meta, dist in zip(retrieved["documents"], retrieved["metadatas"], retrieved["distances"]):
        similarity = 1 - (dist / 2)  # cosine distance -> similarity 0-1
        if similarity >= settings.SIMILARITY_THRESHOLD:
            sources.append(RetrievedTicket(
                text=doc,
                priority=meta.get("priority", "unknown"),
                similarity=round(similarity, 4)
            ))
    
    # 2. Build prompts
    context = "\n\n".join([f"Past Ticket: {s.text}" for s in sources[:3]])
    rag_prompt = f"""Use the following past support tickets to help answer the new question.
    
Past Tickets:
{context}

New Question: {query}

Answer:"""
    non_rag_prompt = f"Answer this customer support question concisely:\n\n{query}"
    llm_priority_prompt = f"""Is the following support ticket urgent or normal? reason.

Reply in exact format:
PREDICTION: [urgent/normal]
CONFIDENCE: [0-100]
REASON: [brief reason]

Ticket: {query}
"""
    
    # 3. Run everything in parallel
    rag_task = asyncio.to_thread(llm.generate, rag_prompt, "You are a helpful customer support assistant. Use the provided past tickets as context.")
    non_rag_task = asyncio.to_thread(llm.generate, non_rag_prompt, "You are a helpful customer support assistant.")
    ml_task = asyncio.to_thread(ml.predict, query)
    llm_priority_task = asyncio.to_thread(llm.generate, llm_priority_prompt, "You are a ticket triage system.")
    
    rag_res, non_rag_res, ml_res, llm_prio_res = await asyncio.gather(
        rag_task, non_rag_task, ml_task, llm_priority_task
    )
    
    # 4. Parse LLM priority
# Find the parsing section (around line 70-75)
 # Parse LLM priority response
    llm_pred_text = llm_prio_res["content"].strip()
    
    # Try structured format first
    pred = "normal"
    confidence = None
    reasoning = llm_pred_text
    
    if "PREDICTION:" in llm_pred_text.upper():
        lines = llm_pred_text.split('\n')
        for line in lines:
            line_upper = line.upper()
            if line_upper.startswith("PREDICTION:"):
                pred = "urgent" if "urgent" in line.lower() else "normal"
            elif line_upper.startswith("CONFIDENCE:"):
                conf_part = line.split(":", 1)[-1].strip()
                # Extract number from string like "85" or "85%"
                import re
                nums = re.findall(r'\d+', conf_part)
                if nums:
                    confidence = float(nums[0]) / 100 if float(nums[0]) > 1 else float(nums[0])
            elif line_upper.startswith("REASON:"):
                reasoning = line.split(":", 1)[-1].strip()
    else:
        # Fallback: extract prediction from free-form text
        text_lower = llm_pred_text.lower()
        if "urgent" in text_lower.split('\n')[0]:
            pred = "urgent"
        else:
            pred = "normal"
        reasoning = llm_pred_text
        
        # Estimate confidence from language
        hedging_words = ['maybe', 'possibly', 'might', 'could be', 'not sure', 'uncertain']
        if any(word in text_lower for word in hedging_words):
            confidence = 0.5
        else:
            confidence = 0.85  # default high confidence for direct statements
    
    # Default confidence if still None
    if confidence is None:
        confidence = 0.85
    
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
            "confidence": round(confidence, 4),  # Now always a float
            "latency_ms": llm_prio_res["latency_ms"],
            "cost_usd": llm_prio_res["cost_usd"]
        }
    )