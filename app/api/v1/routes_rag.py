"""
RAG (Retrieval-Augmented Generation) API Routes
"""

from fastapi import APIRouter, HTTPException
from app.services import get_rag_service
from app.db import supabase
from app.models.query import RAGQueryRequest
from app.core.logging import logger, log_error

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/search")
async def rag_search(request: RAGQueryRequest):
    """
    Perform semantic search with RAG.

    Retrieves relevant context from embeddings, memory, graph, and insights.
    """
    try:
        logger.info(f"RAG search: {request.query[:50]}...")

        rag_service = get_rag_service(supabase)

        context_items = rag_service.retrieve_context(
            query=request.query,
            org_id=str(request.org_id),
            limit=request.limit
        )

        return {
            "success": True,
            "query": request.query,
            "results": context_items,
            "count": len(context_items)
        }

    except Exception as e:
        log_error(e, context="RAG search")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def rag_generate(request: RAGQueryRequest):
    """
    Generate answer using RAG.

    Retrieves context and generates an LLM response.
    """
    try:
        logger.info(f"RAG generate: {request.query[:50]}...")

        rag_service = get_rag_service(supabase)

        response = rag_service.generate_response(
            query=request.query,
            org_id=str(request.org_id),
            limit=request.limit
        )

        return {
            "success": True,
            "query": request.query,
            "answer": response.get("answer"),
            "context": response.get("context"),
            "sources": response.get("sources")
        }

    except Exception as e:
        log_error(e, context="RAG generate")
        raise HTTPException(status_code=500, detail=str(e))
