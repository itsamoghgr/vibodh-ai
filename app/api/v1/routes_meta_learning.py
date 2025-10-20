"""
Meta-Learning API Routes - Phase 3, Step 4
Endpoints for meta-learning, knowledge evolution, and pattern discovery
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from pydantic import BaseModel

from app.services.meta_learning_service import get_meta_learning_service
from app.services.kg_service import get_kg_service
from app.db import supabase, supabase_admin
from app.core.logging import logger

router = APIRouter(prefix="/meta-learning", tags=["Meta-Learning"])


# ===== Request/Response Models =====

class AnalyzeRequest(BaseModel):
    org_id: str
    days_back: Optional[int] = 30


class ApproveSchemaChangeRequest(BaseModel):
    org_id: str
    evolution_id: str
    approved_by: Optional[str] = None


# ===== Endpoints =====

@router.post("/analyze")
async def analyze_meta_learning(request: AnalyzeRequest):
    """
    Trigger meta-learning analysis for an organization.
    Analyzes patterns, generates rules, and detects trends.
    """
    try:
        logger.info(f"Meta-learning analysis requested for org {request.org_id}")

        meta_learning_service = get_meta_learning_service(supabase_admin)

        # Run full meta-knowledge generation
        result = meta_learning_service.generate_meta_knowledge(request.org_id)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Analysis failed'))

        return {
            "success": True,
            "message": "Meta-learning analysis completed",
            "patterns_analyzed": result['patterns'].get('total_patterns', 0),
            "rules_discovered": result['rules_discovered'],
            "trends_detected": bool(result['trends'].get('success')),
            "timestamp": result['timestamp']
        }

    except Exception as e:
        logger.error(f"Meta-learning analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rules")
async def get_meta_rules(
    org_id: str = Query(..., description="Organization ID"),
    days_back: int = Query(30, description="Number of days to look back"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get discovered meta-rules for an organization."""
    try:
        query = supabase.table('ai_meta_knowledge')\
            .select('*')\
            .eq('org_id', org_id)\
            .order('success_rate', desc=True)\
            .order('confidence', desc=True)\
            .limit(50)

        if category:
            query = query.eq('category', category)

        result = query.execute()

        return {
            "success": True,
            "rules": result.data or [],
            "total": len(result.data or [])
        }

    except Exception as e:
        logger.error(f"Get meta rules error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_reasoning_patterns(
    org_id: str = Query(..., description="Organization ID"),
    days_back: int = Query(30, description="Number of days to analyze")
):
    """Get reasoning pattern statistics."""
    try:
        meta_learning_service = get_meta_learning_service(supabase_admin)

        patterns = meta_learning_service.analyze_reasoning_patterns(
            org_id=org_id,
            days_back=days_back
        )

        if not patterns['success']:
            raise HTTPException(status_code=500, detail=patterns.get('error', 'Analysis failed'))

        return {
            "success": True,
            "patterns": patterns['patterns'],
            "intent_patterns": patterns['intent_patterns'],
            "best_combinations": patterns['best_combinations'],
            "total_patterns": patterns['total_patterns']
        }

    except Exception as e:
        logger.error(f"Get reasoning patterns error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/kg-evolution")
async def get_kg_evolution_history(
    org_id: str = Query(..., description="Organization ID"),
    status: Optional[str] = Query(None, description="Filter by status (proposed/approved/rejected/applied)")
):
    """Get Knowledge Graph schema evolution history."""
    try:
        query = supabase.table('kg_schema_evolution')\
            .select('*')\
            .eq('org_id', org_id)\
            .order('created_at', desc=True)\
            .limit(50)

        if status:
            query = query.eq('status', status)

        result = query.execute()

        return {
            "success": True,
            "evolutions": result.data or [],
            "total": len(result.data or [])
        }

    except Exception as e:
        logger.error(f"Get KG evolution history error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kg-evolution/propose")
async def propose_kg_schema_changes(request: AnalyzeRequest):
    """Analyze KG and propose new entity/relation types."""
    try:
        logger.info(f"KG schema evolution proposal for org {request.org_id}")

        kg_service = get_kg_service(supabase_admin)

        # Propose new entity types
        entity_proposals = kg_service.propose_new_entity_types(
            org_id=request.org_id,
            days_back=request.days_back
        )

        # Propose new relation types
        relation_proposals = kg_service.propose_new_relation_types(
            org_id=request.org_id,
            days_back=request.days_back
        )

        return {
            "success": True,
            "entity_type_proposals": entity_proposals,
            "relation_type_proposals": relation_proposals,
            "total_proposals": len(entity_proposals) + len(relation_proposals)
        }

    except Exception as e:
        logger.error(f"KG schema proposal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kg-evolution/approve")
async def approve_schema_change(request: ApproveSchemaChangeRequest):
    """Approve and apply a proposed schema change."""
    try:
        logger.info(f"Approving schema evolution {request.evolution_id} for org {request.org_id}")

        kg_service = get_kg_service(supabase_admin)

        success = kg_service.apply_schema_evolution(
            org_id=request.org_id,
            evolution_id=request.evolution_id,
            approved_by=request.approved_by
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to apply schema evolution")

        return {
            "success": True,
            "message": "Schema evolution approved and applied",
            "evolution_id": request.evolution_id
        }

    except Exception as e:
        logger.error(f"Approve schema change error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kg-evolution/reject")
async def reject_schema_change(request: ApproveSchemaChangeRequest):
    """Reject a proposed schema change."""
    try:
        logger.info(f"Rejecting schema evolution {request.evolution_id} for org {request.org_id}")

        from datetime import datetime

        # Update status to rejected
        supabase_admin.table('kg_schema_evolution')\
            .update({
                'status': 'rejected',
                'reviewed_by': request.approved_by,
                'reviewed_at': datetime.now().isoformat()
            })\
            .eq('id', request.evolution_id)\
            .eq('org_id', request.org_id)\
            .execute()

        return {
            "success": True,
            "message": "Schema evolution rejected",
            "evolution_id": request.evolution_id
        }

    except Exception as e:
        logger.error(f"Reject schema change error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/snapshots")
async def get_model_snapshots(
    org_id: str = Query(..., description="Organization ID"),
    limit: int = Query(20, description="Maximum number of snapshots to return")
):
    """Get model configuration snapshots."""
    try:
        result = supabase.table('ai_model_snapshots')\
            .select('*')\
            .eq('org_id', org_id)\
            .order('created_at', desc=True)\
            .limit(limit)\
            .execute()

        return {
            "success": True,
            "snapshots": result.data or [],
            "total": len(result.data or [])
        }

    except Exception as e:
        logger.error(f"Get model snapshots error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/snapshot/create")
async def create_model_snapshot(
    org_id: str = Query(..., description="Organization ID"),
    snapshot_type: str = Query("manual", description="Snapshot type"),
    notes: Optional[str] = Query(None, description="Optional notes")
):
    """Create a new model configuration snapshot."""
    try:
        meta_learning_service = get_meta_learning_service(supabase_admin)

        snapshot_id = meta_learning_service.create_model_snapshot(
            org_id=org_id,
            snapshot_type=snapshot_type,
            notes=notes
        )

        if not snapshot_id:
            raise HTTPException(status_code=500, detail="Failed to create snapshot")

        return {
            "success": True,
            "message": "Model snapshot created",
            "snapshot_id": snapshot_id
        }

    except Exception as e:
        logger.error(f"Create model snapshot error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_data_trends(
    org_id: str = Query(..., description="Organization ID"),
    days_back: int = Query(30, description="Number of days to analyze")
):
    """Get detected trends in organizational data."""
    try:
        meta_learning_service = get_meta_learning_service(supabase_admin)

        trends = meta_learning_service.detect_trends_in_data(
            org_id=org_id,
            days_back=days_back
        )

        if not trends['success']:
            raise HTTPException(status_code=500, detail=trends.get('error', 'Trend detection failed'))

        return {
            "success": True,
            "trends_analysis": trends['trends_analysis'],
            "entity_growth": trends['entity_growth'],
            "recent_entities_count": len(trends.get('recent_entities', [])),
            "recent_docs_count": trends.get('recent_docs_count', 0)
        }

    except Exception as e:
        logger.error(f"Get data trends error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema-version")
async def get_kg_schema_version(
    org_id: str = Query(..., description="Organization ID")
):
    """Get current Knowledge Graph schema version."""
    try:
        kg_service = get_kg_service(supabase_admin)

        schema_version = kg_service.get_schema_version(org_id)

        return {
            "success": True,
            **schema_version
        }

    except Exception as e:
        logger.error(f"Get schema version error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
