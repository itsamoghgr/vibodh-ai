"""
Document and Embedding Management API Routes
"""

from fastapi import APIRouter, Query, HTTPException

from app.db import supabase
from app.core.logging import logger, log_error
from app.services.ingestion_service import get_ingestion_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("")
async def list_documents(org_id: str = Query(...), limit: int = 100):
    """List documents for an organization"""
    try:
        result = supabase.table("documents")\
            .select("*")\
            .eq("org_id", org_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()

        return {"documents": result.data, "total": len(result.data)}
    except Exception as e:
        log_error(e, context="List documents")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")


@router.post("/retry-failed-embeddings/{org_id}")
async def retry_failed_embeddings(org_id: str):
    """
    Retry embedding generation for all documents with 'failed' status
    First check if embeddings exist, if so just mark as completed
    """
    try:
        ingestion_service = get_ingestion_service(supabase)

        # Get all failed documents
        failed_result = supabase.table("documents")\
            .select("id, content")\
            .eq("org_id", org_id)\
            .eq("embedding_status", "failed")\
            .execute()

        if not failed_result.data:
            return {
                "success": True,
                "message": "No failed embeddings to retry",
                "retried": 0
            }

        success_count = 0
        already_embedded_count = 0
        failed_count = 0

        for doc in failed_result.data:
            try:
                # Check if embeddings already exist for this document
                existing_emb = supabase.table("embeddings")\
                    .select("id")\
                    .eq("org_id", org_id)\
                    .eq("document_id", doc["id"])\
                    .limit(1)\
                    .execute()

                if existing_emb.data and len(existing_emb.data) > 0:
                    # Embeddings exist, just update status
                    supabase.table("documents")\
                        .update({"embedding_status": "completed"})\
                        .eq("id", doc["id"])\
                        .execute()
                    already_embedded_count += 1
                    logger.info(f"Document {doc['id']} already has embeddings, marked as completed")
                else:
                    # No embeddings, generate them
                    await ingestion_service._generate_embeddings(
                        document_id=doc["id"],
                        content=doc["content"],
                        org_id=org_id
                    )
                    success_count += 1
                    logger.info(f"Successfully embedded document {doc['id']}")

            except Exception as e:
                logger.error(f"Failed to embed document {doc['id']}: {str(e)}")
                failed_count += 1

        return {
            "success": True,
            "message": f"Processed {len(failed_result.data)} failed documents",
            "newly_embedded": success_count,
            "already_had_embeddings": already_embedded_count,
            "failed": failed_count,
            "total_processed": len(failed_result.data)
        }

    except Exception as e:
        log_error(e, context="Retry failed embeddings")
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")
