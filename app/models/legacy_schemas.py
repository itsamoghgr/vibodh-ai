# -*- coding: utf-8 -*-
"""
Legacy Pydantic models for API request/response validation
TODO: Migrate these to proper modular structure in app/models/
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================
# SLACK MODELS
# ============================================

class SlackIngestRequest(BaseModel):
    org_id: str
    connection_id: str
    channel_ids: Optional[List[str]] = None  # If None, ingest all channels
    days_back: int = Field(default=30, ge=1, le=365)


# ============================================
# CHAT SESSION MODELS
# ============================================

class ChatStreamRequest(BaseModel):
    query: str
    org_id: str
    user_id: Optional[str] = None  # Optional for memory personalization
    session_id: Optional[str] = None  # Create new session if not provided
    max_context_items: int = Field(default=5, ge=1, le=10)


class ChatSessionResponse(BaseModel):
    id: str
    org_id: str
    user_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: Optional[List[Dict[str, Any]]] = []


class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    tokens: Optional[int]
    context: Optional[List[Dict[str, Any]]]
    created_at: datetime


class FeedbackCreate(BaseModel):
    message_id: str
    rating: str  # 'positive' or 'negative'
    comment: Optional[str] = None
