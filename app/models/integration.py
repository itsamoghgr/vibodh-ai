"""Integration models"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID


class SlackConnectionRequest(BaseModel):
    """Slack connection request"""
    code: str = Field(..., description="OAuth authorization code")
    org_id: UUID = Field(..., description="Organization ID")


class ClickUpConnectionRequest(BaseModel):
    """ClickUp connection request"""
    code: str = Field(..., description="OAuth authorization code")
    org_id: UUID = Field(..., description="Organization ID")


class IntegrationResponse(BaseModel):
    """Integration response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class SyncRequest(BaseModel):
    """Sync request"""
    org_id: UUID = Field(..., description="Organization ID")
    force: bool = Field(False, description="Force full resync")


class SyncResponse(BaseModel):
    """Sync response"""
    success: bool
    message: str
    synced_count: int = 0
    failed_count: int = 0
    details: Optional[Dict[str, Any]] = None
