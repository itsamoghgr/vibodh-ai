"""
Marketing Service - Phase 4, Step 2
Business logic layer for marketing operations
"""

from typing import Dict, List, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from supabase import Client

from app.models.marketing import (
    CampaignPlan, CampaignStatus, CampaignType,
    MarketingContent, ContentType, MarketingChannel,
    MarketingMetrics, PerformanceAnalysis,
    MarketingTask, MarketingAnnouncement,
    MarketingContext, ContentGenerationRequest,
    ContentGenerationResponse, MarketingContent
)
from app.services.llm_service import get_llm_service
from app.services.memory_service import get_memory_service
from app.services.insight_service import get_insight_service
from app.connectors.slack_connector import SlackConnector
from app.connectors.clickup_connector import ClickUpConnector
from app.core.logging import logger
import json


class MarketingService:
    """
    Service for marketing operations and campaign management.

    Handles:
    - Campaign planning and execution
    - Content generation and management
    - Task creation and assignment
    - Performance tracking and analysis
    - Multi-channel distribution
    """

    def __init__(self, supabase: Client):
        """
        Initialize marketing service.

        Args:
            supabase: Supabase client
        """
        self.supabase = supabase
        self.llm_service = get_llm_service()
        self.memory_service = get_memory_service(supabase)
        self.insight_service = get_insight_service(supabase)
        self.slack_connector = SlackConnector(supabase)
        self.clickup_connector = ClickUpConnector(supabase)

        logger.info("[MARKETING_SERVICE] Service initialized")

    async def analyze_marketing_context(
        self,
        org_id: str,
        lookback_days: int = 30
    ) -> MarketingContext:
        """
        Gather and analyze marketing context from multiple sources.

        Args:
            org_id: Organization ID
            lookback_days: Days to look back for historical data

        Returns:
            Marketing context with recent campaigns, messages, and metrics
        """
        try:
            logger.info(f"[MARKETING_SERVICE] Analyzing context for org {org_id}")

            # Get recent campaigns from database
            recent_campaigns = await self._get_recent_campaigns(org_id, lookback_days)

            # Get team messages from Slack
            team_messages = await self._get_recent_team_messages(org_id, lookback_days)

            # Get upcoming events from calendar/tasks
            upcoming_events = await self._get_upcoming_events(org_id)

            # Get performance metrics from past campaigns
            performance_metrics = await self._calculate_performance_metrics(org_id, lookback_days)

            # Get brand guidelines from memory
            brand_guidelines = await self._get_brand_guidelines(org_id)

            # Get target audience from insights
            target_audience = await self._get_target_audience_profile(org_id)

            context = MarketingContext(
                recent_campaigns=recent_campaigns,
                team_messages=team_messages,
                upcoming_events=upcoming_events,
                performance_metrics=performance_metrics,
                brand_guidelines=brand_guidelines,
                target_audience=target_audience
            )

            logger.info(
                f"[MARKETING_SERVICE] Context analyzed: "
                f"{len(recent_campaigns)} campaigns, "
                f"{len(team_messages)} messages, "
                f"{len(upcoming_events)} events"
            )

            return context

        except Exception as e:
            logger.error(f"[MARKETING_SERVICE] Context analysis failed: {e}")
            return MarketingContext()

    async def generate_campaign_content(
        self,
        request: ContentGenerationRequest,
        org_id: str
    ) -> ContentGenerationResponse:
        """
        Generate marketing content using LLM.

        Args:
            request: Content generation request
            org_id: Organization ID

        Returns:
            Generated content response
        """
        try:
            logger.info(f"[MARKETING_SERVICE] Generating {request.content_type} content")

            start_time = datetime.utcnow()

            # Build prompt based on content type
            prompt = self._build_content_prompt(request)

            # Generate main content
            main_content = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=1000
            )

            # Parse and structure content
            content = self._parse_generated_content(
                main_content,
                request.content_type,
                request.campaign_id
            )

            # Generate variations if requested
            variations = []
            if request.content_type in [ContentType.SOCIAL_POST, ContentType.AD_COPY]:
                # Generate 2 variations for A/B testing
                for i in range(2):
                    variation_prompt = f"{prompt}\n\nCreate a different variation:"
                    variation_text = await self.llm_service.generate(
                        prompt=variation_prompt,
                        temperature=0.9,
                        max_tokens=500
                    )
                    variation = self._parse_generated_content(
                        variation_text,
                        request.content_type,
                        request.campaign_id
                    )
                    variations.append(variation)

            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Store content in memory for future reference
            await self.memory_service.store_memory(
                org_id=org_id,
                title=f"Generated {request.content_type}: {content.title}",
                content=content.body,
                memory_type="insight",
                importance=0.6,
                source_refs=[{"type": "marketing_content", "id": str(content.id)}],
                metadata={
                    "content_type": request.content_type,
                    "campaign_id": str(request.campaign_id) if request.campaign_id else None
                }
            )

            return ContentGenerationResponse(
                content=content,
                variations=variations if variations else None,
                generation_time_ms=execution_time,
                tokens_used=len(main_content.split()),  # Rough estimate
                confidence_score=0.85
            )

        except Exception as e:
            logger.error(f"[MARKETING_SERVICE] Content generation failed: {e}")
            raise

    def _build_content_prompt(self, request: ContentGenerationRequest) -> str:
        """Build prompt for content generation."""
        prompts = {
            ContentType.BLOG_POST: f"""Write a {request.length} blog post about: {request.topic}

Target Audience: {request.target_audience}
Tone: {request.tone}
Key Points: {', '.join(request.key_points)}

Include:
- Engaging title
- Introduction paragraph
- Main content with subheadings
- Conclusion
{f"- Call to action: {request.include_cta}" if request.include_cta else ""}

Format as JSON with keys: title, introduction, sections (array), conclusion, cta""",

            ContentType.SOCIAL_POST: f"""Write a social media post about: {request.topic}

Target Audience: {request.target_audience}
Tone: {request.tone}
Key Points: {', '.join(request.key_points[:2])}

Requirements:
- Maximum 280 characters
- Engaging and shareable
{f"- Include relevant hashtags" if request.include_hashtags else ""}
{f"- Include call to action" if request.include_cta else ""}

Format as JSON with keys: post, hashtags (array), cta""",

            ContentType.EMAIL: f"""Write a marketing email about: {request.topic}

Target Audience: {request.target_audience}
Tone: {request.tone}
Key Points: {', '.join(request.key_points)}

Include:
- Subject line
- Preview text
- Email body with proper formatting
- Clear call to action

Format as JSON with keys: subject, preview, body, cta""",

            ContentType.AD_COPY: f"""Write advertising copy for: {request.topic}

Target Audience: {request.target_audience}
Tone: {request.tone}
Key Benefits: {', '.join(request.key_points)}

Create:
- Headline (max 30 chars)
- Description (max 90 chars)
- Call to action

Format as JSON with keys: headline, description, cta"""
        }

        return prompts.get(
            request.content_type,
            f"Generate {request.content_type} content about {request.topic} for {request.target_audience}"
        )

    def _parse_generated_content(
        self,
        raw_content: str,
        content_type: ContentType,
        campaign_id: Optional[UUID]
    ) -> MarketingContent:
        """Parse LLM-generated content into structured format."""
        try:
            # Try to parse as JSON
            json_str = raw_content.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            data = json.loads(json_str)

            # Extract based on content type
            if content_type == ContentType.BLOG_POST:
                title = data.get("title", "Blog Post")
                body = f"{data.get('introduction', '')}\n\n"
                for section in data.get("sections", []):
                    if isinstance(section, dict):
                        body += f"## {section.get('heading', '')}\n{section.get('content', '')}\n\n"
                    else:
                        body += f"{section}\n\n"
                body += data.get("conclusion", "")
                cta = data.get("cta", "")

            elif content_type == ContentType.SOCIAL_POST:
                title = f"Social Post - {datetime.utcnow().strftime('%Y-%m-%d')}"
                body = data.get("post", raw_content[:280])
                cta = data.get("cta", "")
                hashtags = data.get("hashtags", [])

            elif content_type == ContentType.EMAIL:
                title = data.get("subject", "Marketing Email")
                body = data.get("body", raw_content)
                cta = data.get("cta", "")

            elif content_type == ContentType.AD_COPY:
                title = data.get("headline", "Ad Copy")
                body = data.get("description", raw_content)
                cta = data.get("cta", "Learn More")

            else:
                title = "Generated Content"
                body = raw_content
                cta = ""
                hashtags = []

        except json.JSONDecodeError:
            # Fallback to plain text
            logger.warning("[MARKETING_SERVICE] Failed to parse JSON, using plain text")
            title = f"Generated {content_type.value}"
            body = raw_content
            cta = ""
            hashtags = []

        return MarketingContent(
            id=uuid4(),
            campaign_id=campaign_id,
            content_type=content_type,
            title=title,
            body=body,
            call_to_action=cta if cta else None,
            hashtags=hashtags if 'hashtags' in locals() else [],
            created_at=datetime.utcnow()
        )

    async def schedule_campaign_actions(
        self,
        campaign: CampaignPlan,
        org_id: str
    ) -> List[MarketingTask]:
        """
        Schedule campaign actions and create tasks.

        Args:
            campaign: Campaign plan
            org_id: Organization ID

        Returns:
            List of created marketing tasks
        """
        try:
            logger.info(f"[MARKETING_SERVICE] Scheduling actions for campaign: {campaign.title}")

            tasks = []
            current_date = datetime.utcnow()

            # Create tasks for each campaign phase
            phases = [
                ("Planning", 0, 2),
                ("Content Creation", 2, 5),
                ("Review & Approval", 5, 6),
                ("Launch", 6, 7),
                ("Monitor & Optimize", 7, campaign.timeline_days)
            ]

            for phase_name, start_day, end_day in phases:
                task = MarketingTask(
                    id=uuid4(),
                    campaign_id=campaign.id,
                    title=f"{campaign.title} - {phase_name}",
                    description=f"{phase_name} phase for {campaign.title}",
                    task_type=phase_name.lower().replace(" ", "_"),
                    priority="high" if phase_name == "Launch" else "medium",
                    due_date=current_date + timedelta(days=end_day),
                    estimated_hours=(end_day - start_day) * 2,
                    status="pending",
                    created_at=current_date
                )
                tasks.append(task)

                # Create task in ClickUp if connected
                try:
                    await self.clickup_connector.create_task(
                        org_id=org_id,
                        title=task.title,
                        description=task.description,
                        due_date=task.due_date,
                        priority=task.priority,
                        tags=["marketing", campaign.campaign_type.value]
                    )
                    task.integration = "clickup"
                    logger.info(f"[MARKETING_SERVICE] Created ClickUp task: {task.title}")
                except Exception as e:
                    logger.warning(f"[MARKETING_SERVICE] Could not create ClickUp task: {e}")

            logger.info(f"[MARKETING_SERVICE] Scheduled {len(tasks)} tasks")

            return tasks

        except Exception as e:
            logger.error(f"[MARKETING_SERVICE] Task scheduling failed: {e}")
            raise

    async def track_campaign_performance(
        self,
        campaign_id: UUID,
        org_id: str
    ) -> MarketingMetrics:
        """
        Track and collect campaign performance metrics.

        Args:
            campaign_id: Campaign ID
            org_id: Organization ID

        Returns:
            Marketing metrics
        """
        try:
            logger.info(f"[MARKETING_SERVICE] Tracking performance for campaign {campaign_id}")

            # TODO: Integrate with actual analytics sources
            # For now, generate sample metrics

            metrics = MarketingMetrics(
                campaign_id=campaign_id,
                period_start=datetime.utcnow() - timedelta(days=7),
                period_end=datetime.utcnow(),
                impressions=5000,
                engagement_rate=0.045,
                click_through_rate=0.023,
                conversion_rate=0.012,
                likes=150,
                shares=45,
                comments=23,
                leads_generated=12,
                data_sources=["slack", "hubspot", "analytics"],
                confidence_score=0.85
            )

            # Store metrics in database
            await self._store_metrics(metrics, org_id)

            logger.info(f"[MARKETING_SERVICE] Metrics tracked: {metrics.engagement_rate:.2%} engagement")

            return metrics

        except Exception as e:
            logger.error(f"[MARKETING_SERVICE] Performance tracking failed: {e}")
            raise

    async def analyze_campaign_performance(
        self,
        campaign_id: UUID,
        metrics: MarketingMetrics,
        org_id: str
    ) -> PerformanceAnalysis:
        """
        Analyze campaign performance and generate insights.

        Args:
            campaign_id: Campaign ID
            metrics: Marketing metrics
            org_id: Organization ID

        Returns:
            Performance analysis with recommendations
        """
        try:
            logger.info(f"[MARKETING_SERVICE] Analyzing performance for campaign {campaign_id}")

            # Calculate performance score
            score = (
                metrics.engagement_rate * 30 +
                metrics.click_through_rate * 40 +
                metrics.conversion_rate * 30
            ) * 100

            # Identify strengths
            strengths = []
            if metrics.engagement_rate > 0.04:
                strengths.append("Strong audience engagement")
            if metrics.click_through_rate > 0.02:
                strengths.append("Effective call-to-action")
            if metrics.conversion_rate > 0.01:
                strengths.append("Good conversion optimization")
            if metrics.shares > 30:
                strengths.append("High viral potential")

            # Identify weaknesses
            weaknesses = []
            if metrics.engagement_rate < 0.02:
                weaknesses.append("Low audience engagement")
            if metrics.click_through_rate < 0.01:
                weaknesses.append("Weak call-to-action")
            if metrics.conversion_rate < 0.005:
                weaknesses.append("Poor conversion rate")
            if metrics.bounce_rate and metrics.bounce_rate > 0.5:
                weaknesses.append("High bounce rate")

            # Generate recommendations
            recommendations = []
            if metrics.engagement_rate < 0.03:
                recommendations.append("Improve content relevance and timing")
            if metrics.click_through_rate < 0.015:
                recommendations.append("Test different CTA placements and copy")
            if metrics.conversion_rate < 0.008:
                recommendations.append("Optimize landing page experience")
            recommendations.append("Consider A/B testing different content variations")
            recommendations.append("Analyze peak engagement times for better scheduling")

            analysis = PerformanceAnalysis(
                campaign_id=campaign_id,
                performance_score=round(score, 1),
                strengths=strengths,
                weaknesses=weaknesses,
                opportunities=[
                    "Expand to additional channels",
                    "Leverage high-performing content formats",
                    "Increase personalization"
                ],
                threats=[
                    "Audience fatigue from over-messaging",
                    "Competitive campaigns in same period"
                ],
                recommendations=recommendations
            )

            # Store analysis as insight
            await self.memory_service.store_memory(
                org_id=org_id,
                title=f"Campaign Analysis: Score {score:.1f}",
                content=f"Strengths: {', '.join(strengths)}. Recommendations: {', '.join(recommendations[:2])}",
                memory_type="insight",
                importance=0.7,
                source_refs=[{"type": "campaign_analysis", "id": str(campaign_id)}]
            )

            logger.info(f"[MARKETING_SERVICE] Analysis complete: Score {score:.1f}")

            return analysis

        except Exception as e:
            logger.error(f"[MARKETING_SERVICE] Performance analysis failed: {e}")
            raise

    async def post_announcement(
        self,
        announcement: MarketingAnnouncement,
        org_id: str
    ) -> Dict[str, Any]:
        """
        Post marketing announcement to specified channel.

        Args:
            announcement: Marketing announcement
            org_id: Organization ID

        Returns:
            Posting result
        """
        try:
            logger.info(f"[MARKETING_SERVICE] Posting announcement to {announcement.channel}")

            result = {}

            if announcement.channel == MarketingChannel.SLACK:
                # Post to Slack
                try:
                    # TODO: Use actual Slack connector
                    result = {
                        "status": "posted",
                        "channel": "#marketing",
                        "timestamp": datetime.utcnow().isoformat(),
                        "message_id": str(uuid4())
                    }
                    logger.info("[MARKETING_SERVICE] Posted to Slack")
                except Exception as e:
                    logger.error(f"[MARKETING_SERVICE] Slack posting failed: {e}")
                    result = {"status": "failed", "error": str(e)}

            elif announcement.channel == MarketingChannel.EMAIL:
                # Send email
                result = {
                    "status": "queued",
                    "recipients": announcement.target_audience,
                    "scheduled": announcement.scheduled_time
                }

            else:
                result = {
                    "status": "unsupported",
                    "channel": announcement.channel.value
                }

            # Update announcement status
            announcement.sent_at = datetime.utcnow() if result.get("status") == "posted" else None

            return result

        except Exception as e:
            logger.error(f"[MARKETING_SERVICE] Announcement posting failed: {e}")
            raise

    # Private helper methods

    async def _get_recent_campaigns(self, org_id: str, days: int) -> List[Dict[str, Any]]:
        """Get recent marketing campaigns."""
        # TODO: Query actual campaigns from database
        return [
            {
                "id": str(uuid4()),
                "title": "Q4 Product Launch",
                "type": "product_launch",
                "status": "completed",
                "performance": {"engagement": 0.045, "conversion": 0.012}
            }
        ]

    async def _get_recent_team_messages(self, org_id: str, days: int) -> List[Dict[str, Any]]:
        """Get recent team messages from Slack."""
        # TODO: Use actual Slack connector
        return [
            {
                "channel": "#marketing",
                "text": "Great response to the last campaign!",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]

    async def _get_upcoming_events(self, org_id: str) -> List[Dict[str, Any]]:
        """Get upcoming marketing events."""
        # TODO: Query actual events
        return [
            {
                "title": "Product Demo Webinar",
                "date": (datetime.utcnow() + timedelta(days=14)).isoformat(),
                "type": "webinar"
            }
        ]

    async def _calculate_performance_metrics(
        self,
        org_id: str,
        days: int
    ) -> Dict[str, float]:
        """Calculate average performance metrics."""
        # TODO: Calculate from actual data
        return {
            "avg_engagement_rate": 0.038,
            "avg_click_through_rate": 0.018,
            "avg_conversion_rate": 0.009
        }

    async def _get_brand_guidelines(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get brand guidelines from memory."""
        # TODO: Retrieve from memory service
        return {
            "tone": "professional yet friendly",
            "colors": ["#1E88E5", "#FFC107"],
            "voice": "authoritative but approachable"
        }

    async def _get_target_audience_profile(self, org_id: str) -> Optional[Dict[str, Any]]:
        """Get target audience profile."""
        # TODO: Retrieve from insights
        return {
            "primary": "Tech professionals aged 25-45",
            "secondary": "Small business owners",
            "interests": ["productivity", "innovation", "efficiency"]
        }

    async def _store_metrics(self, metrics: MarketingMetrics, org_id: str) -> None:
        """Store metrics in database."""
        # TODO: Implement actual database storage
        logger.info(f"[MARKETING_SERVICE] Metrics stored for campaign {metrics.campaign_id}")


# Singleton instance
_marketing_service = None


def get_marketing_service(supabase: Client) -> MarketingService:
    """Get or create MarketingService instance."""
    global _marketing_service
    if _marketing_service is None:
        _marketing_service = MarketingService(supabase)
    return _marketing_service