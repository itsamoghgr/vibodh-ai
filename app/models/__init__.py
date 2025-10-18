"""Models module - Pydantic data models"""

from .document import DocumentBase, DocumentCreate, DocumentUpdate, DocumentResponse
from .query import QueryRequest, RAGQueryRequest, QueryResponse
from .integration import (
    SlackConnectionRequest,
    ClickUpConnectionRequest,
    IntegrationResponse,
    SyncRequest,
    SyncResponse,
)

__all__ = [
    # Document models
    "DocumentBase",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    # Query models
    "QueryRequest",
    "RAGQueryRequest",
    "QueryResponse",
    # Integration models
    "SlackConnectionRequest",
    "ClickUpConnectionRequest",
    "IntegrationResponse",
    "SyncRequest",
    "SyncResponse",
]
