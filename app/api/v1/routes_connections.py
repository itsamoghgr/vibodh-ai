"""
Connection Management API Routes
"""

from fastapi import APIRouter, Query, HTTPException

from app.db import supabase
from app.core.logging import logger, log_error

router = APIRouter(prefix="/connections", tags=["Connections"])


@router.get("")
async def list_connections(org_id: str = Query(...)):
    """List all connections for an organization"""
    try:
        result = supabase.table("connections")\
            .select("*")\
            .eq("org_id", org_id)\
            .execute()

        return {"connections": result.data}
    except Exception as e:
        log_error(e, context="List connections")
        raise HTTPException(status_code=500, detail=f"Failed to fetch connections: {str(e)}")


@router.delete("/{connection_id}")
async def delete_connection(connection_id: str):
    """Delete a connection"""
    try:
        supabase.table("connections")\
            .delete()\
            .eq("id", connection_id)\
            .execute()

        return {"message": "Connection deleted successfully"}
    except Exception as e:
        log_error(e, context="Delete connection")
        raise HTTPException(status_code=500, detail=f"Failed to delete connection: {str(e)}")
