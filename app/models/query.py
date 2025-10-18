"""Query models"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., min_length=1, description="User query")
    org_id: UUID = Field(..., description="Organization ID")
    user_id: Optional[UUID] = Field(None, description="User ID")


class RAGQueryRequest(QueryRequest):
    """RAG query request model"""
    limit: int = Field(5, ge=1, le=20, description="Number of results")
    include_insights: bool = Field(True, description="Include insights in context")
    include_graph: bool = Field(True, description="Include knowledge graph in context")


class QueryResponse(BaseModel):
    """Query response model"""
    query: str
    intent: Optional[str] = None
    modules_used: Optional[List[str]] = None
    final_answer: str
    context_sources: Optional[List[Dict[str, Any]]] = None
    execution_time_ms: Optional[int] = None
