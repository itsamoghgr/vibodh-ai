# -*- coding: utf-8 -*-
"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================
# BASIC MODELS
# ============================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database: str
    version: str


class OrganizationInfo(BaseModel):
    id: str
    name: str
    created_at: str
    member_count: int


# ============================================
# CONNECTION MODELS
# ============================================

class ConnectionCreate(BaseModel):
    source_type: str
    access_token: str
    refresh_token: Optional[str] = None
    workspace_name: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ConnectionResponse(BaseModel):
    id: str
    org_id: str
    source_type: str
    status: str
    workspace_name: Optional[str]
    connected_at: datetime
    last_sync_at: Optional[datetime]


# ============================================
# DOCUMENT MODELS
# ============================================

class DocumentCreate(BaseModel):
    org_id: str
    connection_id: str
    source_type: str
    source_id: str
    title: Optional[str]
    content: str
    author: Optional[str]
    author_id: Optional[str]
    channel_name: Optional[str]
    channel_id: Optional[str]
    url: Optional[str]
    metadata: Dict[str, Any] = {}


class DocumentResponse(BaseModel):
    id: str
    org_id: str
    source_type: str
    title: Optional[str]
    author: Optional[str]
    channel_name: Optional[str]
    created_at: datetime
    embedding_status: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


# ============================================
# EMBEDDING MODELS
# ============================================

class EmbeddingSearchRequest(BaseModel):
    query: str
    org_id: str
    limit: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class EmbeddingSearchResult(BaseModel):
    document_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any]


class EmbeddingSearchResponse(BaseModel):
    results: List[EmbeddingSearchResult]
    query: str


# ============================================
# INGESTION MODELS
# ============================================

class IngestionJobCreate(BaseModel):
    org_id: str
    connection_id: str
    source_type: str


class IngestionJobResponse(BaseModel):
    id: str
    org_id: str
    source_type: str
    status: str
    documents_fetched: int
    documents_created: int
    embeddings_generated: int
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


# ============================================
# SLACK MODELS
# ============================================

class SlackOAuthCallback(BaseModel):
    code: str
    state: Optional[str] = None


class SlackIngestRequest(BaseModel):
    org_id: str
    connection_id: str
    channel_ids: Optional[List[str]] = None  # If None, ingest all channels
    days_back: int = Field(default=30, ge=1, le=365)


# ============================================
# RAG / CHAT MODELS
# ============================================

class ChatQueryRequest(BaseModel):
    query: str
    org_id: str
    max_context_items: int = Field(default=5, ge=1, le=10)


class ChatQueryResponse(BaseModel):
    query: str
    context: List[EmbeddingSearchResult]
    suggested_answer: Optional[str] = None


# ============================================
# CHAT SESSION MODELS (Step 3)
# ============================================

class ChatSessionCreate(BaseModel):
    org_id: str
    user_id: str
    title: Optional[str] = None


class ChatSessionResponse(BaseModel):
    id: str
    org_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime


class ChatMessageCreate(BaseModel):
    session_id: str
    role: str
    content: str
    tokens: Optional[int] = None
    context: Optional[List[Dict[str, Any]]] = None


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    tokens: Optional[int]
    context: Optional[List[Dict[str, Any]]]
    created_at: datetime


class ChatStreamRequest(BaseModel):
    query: str
    org_id: str
    user_id: Optional[str] = None  # Optional for memory personalization
    session_id: Optional[str] = None  # Create new session if not provided
    max_context_items: int = Field(default=5, ge=1, le=10)


class FeedbackCreate(BaseModel):
    message_id: str
    rating: str  # 'positive' or 'negative'
    comment: Optional[str] = None
