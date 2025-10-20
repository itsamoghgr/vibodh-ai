"""
Agent Models - Phase 4, Step 1
Pydantic models for agent-related data structures
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from enum import Enum


# Enums
class AgentStatus(str, Enum):
    """Agent registry status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"


class PlanStatus(str, Enum):
    """Action plan status"""
    DRAFT = "draft"
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TriggerType(str, Enum):
    """Plan trigger types"""
    MANUAL = "manual"
    EVENT = "event"
    SCHEDULED = "scheduled"
    ORCHESTRATOR = "orchestrator"


class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionStatus(str, Enum):
    """Action execution status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# Agent Configuration Models
class AgentConfig(BaseModel):
    """Agent configuration and metadata"""
    agent_type: str = Field(..., description="Unique agent type identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    description: Optional[str] = Field(None, description="Agent description")
    status: AgentStatus = Field(AgentStatus.ACTIVE, description="Agent status")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    required_permissions: List[str] = Field(default_factory=list, description="Required permissions")
    supported_integrations: List[str] = Field(default_factory=list, description="Supported integrations")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")
    default_risk_threshold: RiskLevel = Field(RiskLevel.MEDIUM, description="Default risk threshold")
    version: str = Field("1.0.0", description="Agent version")


class AgentRegistryEntry(AgentConfig):
    """Agent registry entry from database"""
    id: UUID
    org_id: UUID
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime


# Action Planning Models
class ActionStepRequest(BaseModel):
    """Request model for action step"""
    action_type: str = Field(..., description="Type of action")
    action_name: str = Field(..., description="Name of action")
    description: str = Field(..., description="Description of action")
    target_integration: Optional[str] = Field(None, description="Target integration")
    target_resource: Dict[str, Any] = Field(default_factory=dict, description="Target resource details")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    risk_level: RiskLevel = Field(RiskLevel.LOW, description="Risk level")
    requires_approval: bool = Field(False, description="Whether approval is required")
    depends_on: List[int] = Field(default_factory=list, description="Step dependencies")
    estimated_duration_ms: Optional[int] = Field(None, description="Estimated duration")


class ActionPlanRequest(BaseModel):
    """Request model for creating action plan"""
    agent_type: str = Field(..., description="Agent type")
    goal: str = Field(..., description="Goal to achieve")
    description: Optional[str] = Field(None, description="Plan description")
    trigger_type: TriggerType = Field(TriggerType.MANUAL, description="Trigger type")
    trigger_source: Dict[str, Any] = Field(default_factory=dict, description="Trigger source details")
    steps: List[ActionStepRequest] = Field(..., min_items=1, description="Plan steps")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ActionPlanResponse(BaseModel):
    """Response model for action plan"""
    id: UUID
    org_id: UUID
    agent_type: str
    goal: str
    description: Optional[str]
    status: PlanStatus
    trigger_type: TriggerType
    trigger_source: Dict[str, Any]
    steps: List[Dict[str, Any]]
    total_steps: int
    completed_steps: int
    risk_level: RiskLevel
    requires_approval: bool
    approval_status: Optional[str]
    approved_by: Optional[UUID]
    approved_at: Optional[datetime]
    rejection_reason: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    execution_time_ms: Optional[int]
    context: Dict[str, Any]
    results: Dict[str, Any]
    error_message: Optional[str]
    initiated_by: Optional[UUID]
    created_at: datetime
    updated_at: datetime


# Action Execution Models
class PendingActionResponse(BaseModel):
    """Response model for pending action"""
    id: UUID
    org_id: UUID
    plan_id: UUID
    step_index: int
    action_type: str
    action_name: str
    description: Optional[str]
    target_integration: Optional[str]
    target_resource: Dict[str, Any]
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    requires_approval: bool
    auto_approve_timeout: Optional[datetime]
    validation_status: str
    validation_errors: List[str]
    depends_on: List[UUID]
    status: ActionStatus
    approved_by: Optional[UUID]
    approved_at: Optional[datetime]
    rejection_reason: Optional[str]
    created_at: datetime
    expires_at: datetime


class ActionApprovalRequest(BaseModel):
    """Request model for action approval"""
    action: str = Field(..., description="approve or reject")
    reason: Optional[str] = Field(None, description="Reason for rejection")


class ExecutedActionResponse(BaseModel):
    """Response model for executed action"""
    id: UUID
    org_id: UUID
    plan_id: Optional[UUID]
    pending_action_id: Optional[UUID]
    agent_type: str
    action_type: str
    action_name: str
    description: Optional[str]
    target_integration: Optional[str]
    target_resource: Dict[str, Any]
    parameters: Dict[str, Any]
    status: ActionStatus
    result: Dict[str, Any]
    error_message: Optional[str]
    error_details: Optional[Dict[str, Any]]
    started_at: datetime
    completed_at: datetime
    execution_time_ms: int
    side_effects: List[Dict[str, Any]]
    can_rollback: bool
    rollback_action_id: Optional[UUID]
    executed_by: Optional[UUID]
    created_at: datetime


# Agent Event Models
class AgentEventType(str, Enum):
    """Agent event types"""
    REGISTERED = "registered"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"
    TRIGGERED = "triggered"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFLECTED = "reflected"


class AgentEvent(BaseModel):
    """Agent event model"""
    agent_type: str = Field(..., description="Agent type")
    event_type: AgentEventType = Field(..., description="Event type")
    event_source: Optional[str] = Field(None, description="Event source")
    plan_id: Optional[UUID] = Field(None, description="Associated plan ID")
    action_id: Optional[UUID] = Field(None, description="Associated action ID")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Event metadata")


class AgentEventResponse(AgentEvent):
    """Response model for agent event"""
    id: UUID
    org_id: UUID
    user_id: Optional[UUID]
    created_at: datetime


# Agent Execution Request/Response
class AgentExecuteRequest(BaseModel):
    """Request model for agent execution"""
    agent_type: str = Field(..., description="Agent type to execute")
    goal: str = Field(..., description="Goal to achieve")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    auto_approve: bool = Field(False, description="Auto-approve low-risk actions")
    dry_run: bool = Field(False, description="Simulate execution without making changes")


class AgentExecuteResponse(BaseModel):
    """Response model for agent execution"""
    plan_id: UUID
    agent_type: str
    goal: str
    status: PlanStatus
    plan_created: bool
    requires_approval: bool
    risk_level: RiskLevel
    total_steps: int
    message: str
    next_action: Optional[str] = None  # What user should do next


# Agent Status Models
class AgentStatusResponse(BaseModel):
    """Response model for agent status"""
    agent_type: str
    agent_name: str
    status: AgentStatus
    state: str
    has_current_plan: bool
    execution_history_count: int
    capabilities: List[str]
    supported_integrations: List[str]
    active_plans_count: Optional[int] = 0
    pending_actions_count: Optional[int] = 0


# Statistics Models
class AgentStatistics(BaseModel):
    """Agent performance statistics"""
    agent_type: str
    total_plans: int
    completed_plans: int
    failed_plans: int
    success_rate: float
    average_execution_time_ms: int
    total_actions_executed: int
    high_risk_actions: int
    approvals_required: int
    approvals_granted: int
    approvals_rejected: int
    last_execution: Optional[datetime]