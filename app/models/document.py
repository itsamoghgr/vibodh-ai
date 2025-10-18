"""Document models"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class DocumentBase(BaseModel):
    """Base document model"""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    source_type: str = Field(..., description="Source type (slack, clickup, manual)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentCreate(DocumentBase):
    """Document creation model"""
    org_id: UUID = Field(..., description="Organization ID")


class DocumentUpdate(BaseModel):
    """Document update model"""
    title: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentResponse(DocumentBase):
    """Document response model"""
    id: UUID
    org_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
