"""
Marketing Models - Phase 4, Step 2
Data structures for marketing agent operations
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from enum import Enum


# Enums
class CampaignType(str, Enum):
    """Types of marketing campaigns"""
    PRODUCT_LAUNCH = "product_launch"
    NEWSLETTER = "newsletter"
    SOCIAL_MEDIA = "social_media"
    EVENT_PROMOTION = "event_promotion"
    EMAIL = "email"
    CONTENT_MARKETING = "content_marketing"
    BRAND_AWARENESS = "brand_awareness"


class CampaignStatus(str, Enum):
    """Campaign execution status"""
    DRAFT = "draft"
    PLANNING = "planning"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ContentType(str, Enum):
    """Types of marketing content"""
    BLOG_POST = "blog_post"
    SOCIAL_POST = "social_post"
    EMAIL = "email"
    NEWSLETTER = "newsletter"
    VIDEO_SCRIPT = "video_script"
    INFOGRAPHIC = "infographic"
    PRESS_RELEASE = "press_release"
    AD_COPY = "ad_copy"


class MarketingChannel(str, Enum):
    """Marketing distribution channels"""
    SLACK = "slack"
    EMAIL = "email"
    SOCIAL_MEDIA = "social_media"
    WEBSITE = "website"
    HUBSPOT = "hubspot"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"


# Marketing Context Models
class MarketingContext(BaseModel):
    """Context for marketing planning"""
    recent_campaigns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent campaign data"
    )
    team_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent team communications"
    )
    upcoming_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Upcoming events or deadlines"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Historical performance data"
    )
    brand_guidelines: Optional[Dict[str, Any]] = Field(
        None,
        description="Brand voice and style guidelines"
    )
    target_audience: Optional[Dict[str, Any]] = Field(
        None,
        description="Target audience demographics"
    )


# Campaign Models
class CampaignPlan(BaseModel):
    """Marketing campaign plan"""
    id: Optional[UUID] = None
    title: str = Field(..., max_length=100, description="Campaign title")
    description: str = Field(..., max_length=500, description="Campaign description")
    campaign_type: CampaignType = Field(..., description="Type of campaign")
    goals: List[str] = Field(..., description="Campaign goals")
    target_audience: str = Field(..., description="Target audience description")
    channels: List[MarketingChannel] = Field(..., description="Distribution channels")
    timeline_days: int = Field(..., ge=1, le=365, description="Campaign duration in days")
    budget: Optional[float] = Field(None, ge=0, description="Campaign budget")
    key_messages: List[str] = Field(..., max_items=5, description="Key messages")
    success_metrics: List[str] = Field(..., description="Success metrics")
    estimated_reach: str = Field(..., description="Estimated audience reach")
    content_plan: Optional[List['ContentPlan']] = Field(None, description="Content items")
    status: CampaignStatus = Field(CampaignStatus.DRAFT, description="Campaign status")
    created_at: Optional[datetime] = None
    launched_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ContentPlan(BaseModel):
    """Plan for a piece of marketing content"""
    content_type: ContentType = Field(..., description="Type of content")
    title: str = Field(..., max_length=200, description="Content title")
    description: str = Field(..., description="Content description")
    channel: MarketingChannel = Field(..., description="Primary distribution channel")
    scheduled_date: Optional[datetime] = Field(None, description="Scheduled publish date")
    keywords: List[str] = Field(default_factory=list, description="SEO/targeting keywords")
    estimated_time_hours: float = Field(1.0, ge=0.1, description="Estimated creation time")
    assigned_to: Optional[str] = Field(None, description="Assigned team member")
    status: str = Field("planned", description="Content status")


# Content Generation Models
class MarketingContent(BaseModel):
    """Generated marketing content"""
    id: Optional[UUID] = None
    campaign_id: Optional[UUID] = None
    content_type: ContentType = Field(..., description="Type of content")
    title: str = Field(..., description="Content title")
    body: str = Field(..., description="Main content body")
    summary: Optional[str] = Field(None, description="Content summary")
    call_to_action: Optional[str] = Field(None, description="CTA text")
    hashtags: List[str] = Field(default_factory=list, description="Social hashtags")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    media_urls: List[str] = Field(default_factory=list, description="Associated media")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    generated_by: str = Field("marketing_agent", description="Generation source")


class ContentGenerationRequest(BaseModel):
    """Request to generate marketing content"""
    campaign_id: Optional[UUID] = None
    content_type: ContentType = Field(..., description="Type of content to generate")
    topic: str = Field(..., description="Content topic or theme")
    tone: str = Field("professional", description="Content tone")
    length: str = Field("medium", description="Content length (short/medium/long)")
    target_audience: str = Field(..., description="Target audience")
    key_points: List[str] = Field(default_factory=list, description="Key points to include")
    brand_voice: Optional[Dict[str, Any]] = Field(None, description="Brand voice guidelines")
    include_cta: bool = Field(True, description="Include call-to-action")
    include_hashtags: bool = Field(False, description="Generate hashtags")


# Performance Tracking Models
class MarketingMetrics(BaseModel):
    """Marketing performance metrics"""
    campaign_id: UUID = Field(..., description="Campaign ID")
    period_start: datetime = Field(..., description="Metrics period start")
    period_end: datetime = Field(..., description="Metrics period end")

    # Engagement metrics
    impressions: int = Field(0, ge=0, description="Total impressions")
    engagement_rate: float = Field(0.0, ge=0.0, le=1.0, description="Engagement rate")
    click_through_rate: float = Field(0.0, ge=0.0, le=1.0, description="CTR")
    conversion_rate: float = Field(0.0, ge=0.0, le=1.0, description="Conversion rate")

    # Social metrics
    likes: int = Field(0, ge=0, description="Total likes")
    shares: int = Field(0, ge=0, description="Total shares")
    comments: int = Field(0, ge=0, description="Total comments")

    # Email metrics
    open_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Email open rate")
    bounce_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Email bounce rate")
    unsubscribe_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Unsubscribe rate")

    # Business metrics
    leads_generated: int = Field(0, ge=0, description="Leads generated")
    revenue_attributed: Optional[float] = Field(None, ge=0, description="Revenue attributed")
    cost_per_acquisition: Optional[float] = Field(None, ge=0, description="CPA")
    return_on_investment: Optional[float] = Field(None, description="ROI")

    # Metadata
    data_sources: List[str] = Field(default_factory=list, description="Data sources")
    confidence_score: float = Field(0.8, ge=0.0, le=1.0, description="Data confidence")
    notes: Optional[str] = Field(None, description="Additional notes")


class PerformanceAnalysis(BaseModel):
    """Analysis of marketing performance"""
    campaign_id: UUID = Field(..., description="Campaign ID")
    analysis_date: datetime = Field(default_factory=datetime.utcnow)
    performance_score: float = Field(..., ge=0.0, le=100.0, description="Overall score")
    strengths: List[str] = Field(default_factory=list, description="Campaign strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Campaign weaknesses")
    opportunities: List[str] = Field(default_factory=list, description="Opportunities")
    threats: List[str] = Field(default_factory=list, description="Threats or risks")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    competitor_comparison: Optional[Dict[str, Any]] = Field(None, description="Competitor data")
    predicted_outcome: Optional[str] = Field(None, description="Predicted campaign outcome")


# Task Management Models
class MarketingTask(BaseModel):
    """Marketing task for team assignment"""
    id: Optional[UUID] = None
    campaign_id: Optional[UUID] = None
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    task_type: str = Field(..., description="Type of task")
    priority: str = Field("medium", description="Task priority")
    assigned_to: Optional[str] = Field(None, description="Assignee")
    due_date: Optional[datetime] = Field(None, description="Due date")
    estimated_hours: float = Field(1.0, ge=0.1, description="Estimated hours")
    dependencies: List[UUID] = Field(default_factory=list, description="Dependent task IDs")
    status: str = Field("pending", description="Task status")
    integration: Optional[str] = Field(None, description="Integration tool (clickup, jira)")
    external_id: Optional[str] = Field(None, description="External system ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# Announcement Models
class MarketingAnnouncement(BaseModel):
    """Marketing announcement or message"""
    id: Optional[UUID] = None
    campaign_id: Optional[UUID] = None
    channel: MarketingChannel = Field(..., description="Distribution channel")
    title: str = Field(..., description="Announcement title")
    message: str = Field(..., description="Message content")
    target_audience: str = Field(..., description="Target audience")
    scheduled_time: Optional[datetime] = Field(None, description="Scheduled send time")
    is_draft: bool = Field(True, description="Draft status")
    requires_approval: bool = Field(False, description="Needs approval")
    approved_by: Optional[str] = Field(None, description="Approver")
    sent_at: Optional[datetime] = Field(None, description="Actual send time")
    engagement_metrics: Optional[Dict[str, Any]] = Field(None, description="Engagement data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional data")


# Response Models
class CampaignExecutionResponse(BaseModel):
    """Response from campaign execution"""
    campaign_id: UUID = Field(..., description="Campaign ID")
    status: str = Field(..., description="Execution status")
    actions_taken: List[Dict[str, Any]] = Field(..., description="Actions executed")
    results: Dict[str, Any] = Field(..., description="Execution results")
    errors: List[str] = Field(default_factory=list, description="Any errors")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    execution_time_ms: int = Field(..., description="Execution time")


class ContentGenerationResponse(BaseModel):
    """Response from content generation"""
    content: MarketingContent = Field(..., description="Generated content")
    variations: Optional[List[MarketingContent]] = Field(None, description="Content variations")
    generation_time_ms: int = Field(..., description="Generation time")
    tokens_used: Optional[int] = Field(None, description="LLM tokens used")
    confidence_score: float = Field(0.8, ge=0.0, le=1.0, description="Content quality score")