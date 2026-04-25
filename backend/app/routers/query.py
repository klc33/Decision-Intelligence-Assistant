import asyncio
import re
from fastapi import APIRouter
from app.schemas import QueryRequest, QueryResponse, RetrievedTicket
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.services.ml_inference import MLInferenceService
from app.config import settings
from app.utils.logging import logger

router = APIRouter()

vector_store = VectorStoreService()
llm = LLMService()
ml = MLInferenceService()

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    query = request.text

    # 1. Retrieve similar chunks from ChromaDB
    retrieved = vector_store.query(query)
    sources = []
    for doc, meta, dist in zip(retrieved["documents"], retrieved["metadatas"], retrieved["distances"]):
        similarity = 1 - (dist / 2)  # cosine distance -> 0-1 similarity
        if similarity >= settings.SIMILARITY_THRESHOLD:
            sources.append(RetrievedTicket(
                text=doc,
                priority=meta.get("priority", "unknown") if meta else "unknown",
                similarity=round(similarity, 4)
            ))

    # 2. Build prompts
    context = "\n\n".join([f"Past Solution Chunk {i+1}: {s.text}" for i, s in enumerate(sources[:3])])

    rag_prompt = f"""You are a customer support assistant. Use the following past support replies to answer the customer's question.

- Read all the past replies carefully.
- Piece together the information to give the best possible advice.
- If the past replies suggest any actions (like contacting support via DM, or checking warranty, or visiting a technician), include them in your answer.
- Do NOT make up information not present in the replies. If absolutely no relevant advice exists, say: "I couldn't find a specific solution in the past replies, but you may need to contact support directly."

PAST SUPPORT REPLIES:
{context}

CUSTOMER QUESTION: {query}

ANSWER:"""
    non_rag_prompt = f"""Answer the customer's question directly. Do NOT ask follow‑up questions. Do NOT ask for more information. Give the best answer you can right now.

QUESTION: {query}

DIRECT ANSWER:"""

    llm_priority_prompt = f"""Is the following support ticket urgent or normal? Reply with only one word: "urgent" or "normal". Then provide a brief reason.

Ticket: {query}
"""

    # 3. Run all tasks in parallel
    rag_task = asyncio.to_thread(
        llm.generate,
        rag_prompt,
        "You are a helpful customer support assistant. Use the provided past ticket chunks as context."
    )
    non_rag_task = asyncio.to_thread(
        llm.generate,
        non_rag_prompt,
        "You are a helpful customer support assistant."
    )
    ml_task = asyncio.to_thread(ml.predict, query)
    llm_priority_task = asyncio.to_thread(
        llm.generate,
        llm_priority_prompt,
        "You are a ticket triage system."
    )

    rag_res, non_rag_res, ml_res, llm_prio_res = await asyncio.gather(
        rag_task, non_rag_task, ml_task, llm_priority_task
    )

    # 4. Parse LLM priority (no confidence)
    llm_pred_text = llm_prio_res["content"].strip().lower()
    pred = "urgent" if "urgent" in llm_pred_text else "normal"
    reasoning = llm_pred_text.split('\n', 1)[-1] if '\n' in llm_pred_text else llm_pred_text
    
    
    logger.info(f"Query received: {query}")
    # ... after retrieval ...
    logger.info(f"Retrieved {len(sources)} sources above threshold")
    # ... after LLM calls ...
    logger.info(f"RAG latency: {rag_res['latency_ms']}ms, cost: {rag_res['cost_usd']}USD")
    logger.info(f"Non-RAG latency: {non_rag_res['latency_ms']}ms, cost: {non_rag_res['cost_usd']}USD")
    logger.info(f"ML priority: {ml_res['prediction']} (confidence: {ml_res['confidence']})")
    logger.info(f"LLM priority: {pred} (latency: {llm_prio_res['latency_ms']}ms, cost: {llm_prio_res['cost_usd']}USD)")

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