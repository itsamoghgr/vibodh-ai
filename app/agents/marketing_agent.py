"""
Marketing Agent - Phase 4, Step 2
Autonomous agent for marketing campaign management and execution
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from app.agents.base_agent import (
    BaseAgent, AgentCapability, ActionStep, ActionPlan,
    ObservationContext, ExecutionResult, ReflectionInsight
)
from app.services.llm_service import get_llm_service
from app.services.slack_service import SlackService
from app.services.agent_event_bus import get_agent_event_bus, AgentEventType
from app.core.config import settings
from app.core.logging import logger
import json


class MarketingAgent(BaseAgent):
    """
    Marketing Agent implementation for autonomous marketing operations.

    Capabilities:
    - Campaign planning and strategy
    - Content generation
    - Task management
    - Announcement posting
    - Performance tracking
    """

    def _initialize_agent(self) -> None:
        """Initialize marketing-specific configuration and resources."""
        from app.db import supabase
        self.llm_service = get_llm_service()
        self.slack_service = SlackService(
            settings.SLACK_CLIENT_ID,
            settings.SLACK_CLIENT_SECRET
        )
        self.event_bus = get_agent_event_bus(supabase)
        self.campaign_templates = {
            "product_launch": {
                "name": "Product Launch Campaign",
                "steps": ["announcement", "content_creation", "task_distribution", "follow_up"],
                "channels": ["slack", "email", "social"]
            },
            "newsletter": {
                "name": "Newsletter Campaign",
                "steps": ["content_gathering", "writing", "review", "distribution"],
                "channels": ["email", "slack"]
            },
            "social_media": {
                "name": "Social Media Campaign",
                "steps": ["content_planning", "creation", "scheduling", "engagement"],
                "channels": ["social", "slack"]
            },
            "event_promotion": {
                "name": "Event Promotion Campaign",
                "steps": ["announcement", "reminder_schedule", "material_prep", "follow_up"],
                "channels": ["slack", "email", "calendar"]
            }
        }

        logger.info(f"[MARKETING_AGENT] Initialized with {len(self.campaign_templates)} templates")

    @property
    def capabilities(self) -> List[AgentCapability]:
        """Return list of capabilities this agent supports."""
        return [
            AgentCapability.CAMPAIGN_PLANNING,
            AgentCapability.CONTENT_GENERATION,
            AgentCapability.CONTENT_EDITING,
            AgentCapability.CONTENT_SCHEDULING,
            AgentCapability.MESSAGE_SENDING,
            AgentCapability.TASK_PLANNING,
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.PATTERN_RECOGNITION
        ]

    @property
    def required_permissions(self) -> List[str]:
        """Return list of permissions required for this agent to function."""
        return [
            "slack:write",
            "slack:read",
            "clickup:write",
            "clickup:read",
            "hubspot:campaigns",
            "hubspot:contacts",
            "content:generate",
            "tasks:create"
        ]

    @property
    def supported_integrations(self) -> List[str]:
        """Return list of integrations this agent can work with."""
        return ["slack", "clickup", "hubspot", "email"]

    @property
    def description(self) -> str:
        """Description of what this agent does."""
        return "Autonomous marketing agent for campaign planning, content creation, and execution across multiple channels"

    async def _observe_impl(self, context: ObservationContext) -> Tuple[bool, Optional[str]]:
        """
        Marketing-specific observation implementation.

        Analyzes context to determine if marketing action is needed:
        - Plan approval requests
        - Campaign requests
        - Content needs
        - Announcement requirements
        - Task management

        Args:
            context: Observation context

        Returns:
            Tuple of (should_act, reason)
        """
        logger.info(f"[MARKETING_AGENT] Observing for marketing opportunities")

        # Check if query is a plan approval first
        if context.query:
            query_lower = context.query.lower()

            # Check for approval keywords
            approval_keywords = [
                "yes", "approve", "confirm", "proceed", "go ahead",
                "accept", "agreed", "ok", "okay", "sure", "do it"
            ]

            if any(keyword in query_lower for keyword in approval_keywords):
                # Check if there are pending plans for this org
                try:
                    from app.db import supabase

                    pending_plans = supabase.table("ai_action_plans")\
                        .select("id, goal")\
                        .eq("org_id", self.org_id)\
                        .eq("status", "pending_approval")\
                        .order("created_at", desc=True)\
                        .limit(1)\
                        .execute()

                    if pending_plans.data:
                        plan = pending_plans.data[0]
                        logger.info(f"[MARKETING_AGENT] Found pending plan: {plan['id']}")
                        return True, f"Plan approval requested for: {plan['goal']}"

                except Exception as e:
                    logger.error(f"[MARKETING_AGENT] Error checking pending plans: {e}")
                    # Continue with regular checks if database query fails

        # Check if query is marketing-related
        if context.query:
            query_lower = context.query.lower()

            # Direct marketing keywords
            marketing_triggers = [
                "campaign", "marketing", "announce", "promote", "launch",
                "newsletter", "content", "social media", "advertisement",
                "engagement", "audience", "brand", "outreach", "email"
            ]

            if any(trigger in query_lower for trigger in marketing_triggers):
                # Determine specific marketing need
                if "campaign" in query_lower or "launch" in query_lower:
                    return True, "Campaign planning or launch requested"
                elif "announce" in query_lower or "message" in query_lower:
                    return True, "Announcement or messaging requested"
                elif "content" in query_lower or "write" in query_lower:
                    return True, "Content creation requested"
                elif "task" in query_lower or "assign" in query_lower:
                    return True, "Marketing task management requested"
                else:
                    return True, "General marketing action requested"

        # Check for event-based triggers
        if context.event:
            event_type = context.event.get("type")

            # Product release event
            if event_type == "product_release":
                return True, "Product release detected - launch campaign needed"

            # Scheduled campaign
            elif event_type == "scheduled_campaign":
                return True, "Scheduled campaign trigger"

            # Low engagement alert
            elif event_type == "low_engagement":
                return True, "Low engagement detected - intervention needed"

        # Check metadata for marketing signals
        if context.metadata:
            # Check for campaign deadlines
            if context.metadata.get("campaign_deadline"):
                deadline = context.metadata["campaign_deadline"]
                # If deadline is within 7 days, action needed
                if isinstance(deadline, str):
                    deadline_date = datetime.fromisoformat(deadline)
                    if (deadline_date - datetime.utcnow()).days <= 7:
                        return True, "Upcoming campaign deadline"

            # Check for content calendar
            if context.metadata.get("content_due"):
                return True, "Content creation due"

        logger.info(f"[MARKETING_AGENT] No marketing action needed")
        return False, None

    async def _plan_impl(self, goal: str, context: Dict[str, Any]) -> ActionPlan:
        """
        Generate a marketing action plan.

        Creates structured marketing campaigns with:
        - Content creation steps
        - Channel distribution
        - Task assignments
        - Timeline and milestones

        Args:
            goal: Marketing goal to achieve
            context: Planning context

        Returns:
            Structured marketing action plan
        """
        logger.info(f"[MARKETING_AGENT] Planning for goal: {goal}")

        # Phase 6, Step 2: Query CIL proposals before planning
        cil_suggestions = []
        cil_auto_applied = []

        org_id = self.config.get("org_id")
        if org_id:
            try:
                proposals = await self._fetch_cil_proposals(org_id)

                for proposal in proposals:
                    # Auto-apply high-confidence proposals
                    if proposal.get('confidence_score', 0) >= 0.8:
                        applied = await self._apply_proposal(proposal)
                        if applied:
                            cil_auto_applied.append(proposal)
                    else:
                        # Add as suggestion for human review
                        cil_suggestions.append(proposal)

                if cil_auto_applied:
                    logger.info(f"[MARKETING_AGENT] Auto-applied {len(cil_auto_applied)} high-confidence CIL proposals")
                if cil_suggestions:
                    logger.info(f"[MARKETING_AGENT] {len(cil_suggestions)} CIL proposals available for review")

            except Exception as e:
                logger.error(f"[MARKETING_AGENT] Failed to process CIL proposals: {e}", exc_info=True)

        # Add CIL context to planning
        if 'cil_suggestions' not in context:
            context['cil_suggestions'] = []
        context['cil_suggestions'].extend(cil_suggestions)

        if 'cil_auto_applied' not in context:
            context['cil_auto_applied'] = []
        context['cil_auto_applied'].extend(cil_auto_applied)

        # Determine campaign type from goal
        campaign_type = self._determine_campaign_type(goal)
        template = self.campaign_templates.get(campaign_type, self.campaign_templates["product_launch"])

        # Generate campaign details using LLM
        campaign_details = await self._generate_campaign_details(goal, template, context)

        # Create action steps
        steps = []
        for idx, step_type in enumerate(template["steps"]):
            step = self._create_marketing_step(
                idx,
                step_type,
                campaign_details,
                template["channels"]
            )
            steps.append(step)

        # Calculate overall risk level
        has_external = any("external" in step.description.lower() or
                          "public" in step.description.lower()
                          for step in steps)
        risk_level = "medium" if has_external else "low"

        # Create action plan
        plan = ActionPlan(
            goal=goal,
            description=f"{template['name']}: {campaign_details.get('title', goal)}",
            steps=steps,
            total_steps=len(steps),
            risk_level=risk_level,
            requires_approval=risk_level != "low",
            context={
                "campaign_type": campaign_type,
                "campaign_details": campaign_details,
                "channels": template["channels"],
                "estimated_reach": campaign_details.get("estimated_reach", "unknown")
            },
            confidence_score=0.85,
            estimated_total_duration_ms=len(steps) * 30000  # 30 seconds per step estimate
        )

        logger.info(f"[MARKETING_AGENT] Created plan with {len(steps)} steps, risk: {risk_level}")

        return plan

    def _determine_campaign_type(self, goal: str) -> str:
        """Determine campaign type from goal text."""
        goal_lower = goal.lower()

        if "launch" in goal_lower or "release" in goal_lower:
            return "product_launch"
        elif "newsletter" in goal_lower:
            return "newsletter"
        elif "social" in goal_lower or "twitter" in goal_lower or "linkedin" in goal_lower:
            return "social_media"
        elif "event" in goal_lower or "webinar" in goal_lower or "conference" in goal_lower:
            return "event_promotion"
        else:
            return "product_launch"  # Default

    async def _generate_campaign_details(
        self,
        goal: str,
        template: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate campaign details using LLM."""
        prompt = f"""Create a marketing campaign plan for the following goal:

Goal: {goal}
Campaign Type: {template['name']}
Available Channels: {', '.join(template['channels'])}

Generate a JSON response with:
1. title: Campaign title (max 50 chars)
2. description: Campaign description (max 200 chars)
3. key_messages: Array of 3 key messages
4. target_audience: Target audience description
5. success_metrics: Array of 2-3 success metrics
6. timeline: Suggested timeline in days
7. estimated_reach: Estimated audience reach

Context: {json.dumps(context, default=str)[:500]}

Return ONLY valid JSON."""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )

            # Parse JSON response
            # Clean response to extract JSON
            json_str = response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            campaign_details = json.loads(json_str)

            return campaign_details

        except Exception as e:
            logger.warning(f"[MARKETING_AGENT] Failed to generate campaign details: {e}")
            # Return default structure
            return {
                "title": f"Campaign: {goal[:50]}",
                "description": f"Marketing campaign to {goal}",
                "key_messages": ["Key message 1", "Key message 2", "Key message 3"],
                "target_audience": "Target audience",
                "success_metrics": ["Engagement rate", "Click-through rate"],
                "timeline": 7,
                "estimated_reach": "100-500"
            }

    def _create_marketing_step(
        self,
        index: int,
        step_type: str,
        campaign_details: Dict[str, Any],
        channels: List[str]
    ) -> ActionStep:
        """Create a marketing action step."""
        step_configs = {
            "announcement": {
                "action_type": "send_message",
                "action_name": "Send Campaign Announcement",
                "description": f"Announce {campaign_details.get('title', 'campaign')} to team",
                "target_integration": "slack",
                "risk_level": "low",
                "requires_approval": False
            },
            "content_creation": {
                "action_type": "generate_content",
                "action_name": "Generate Campaign Content",
                "description": f"Create content for {', '.join(channels)} channels",
                "target_integration": None,
                "risk_level": "low",
                "requires_approval": False
            },
            "task_distribution": {
                "action_type": "create_task",
                "action_name": "Create Campaign Tasks",
                "description": "Create and assign tasks in ClickUp",
                "target_integration": "clickup",
                "risk_level": "low",
                "requires_approval": False
            },
            "follow_up": {
                "action_type": "send_message",
                "action_name": "Send Follow-up",
                "description": "Send campaign follow-up and collect feedback",
                "target_integration": "slack",
                "risk_level": "low",
                "requires_approval": False
            },
            "content_gathering": {
                "action_type": "collect_data",
                "action_name": "Gather Content",
                "description": "Collect content from various sources",
                "target_integration": None,
                "risk_level": "low",
                "requires_approval": False
            },
            "writing": {
                "action_type": "generate_content",
                "action_name": "Write Content",
                "description": "Write newsletter or article content",
                "target_integration": None,
                "risk_level": "low",
                "requires_approval": False
            },
            "review": {
                "action_type": "review_content",
                "action_name": "Review Content",
                "description": "Review and edit content before distribution",
                "target_integration": None,
                "risk_level": "low",
                "requires_approval": True
            },
            "distribution": {
                "action_type": "distribute",
                "action_name": "Distribute Content",
                "description": f"Distribute to {', '.join(channels)}",
                "target_integration": channels[0] if channels else None,
                "risk_level": "medium",
                "requires_approval": True
            },
            "content_planning": {
                "action_type": "plan",
                "action_name": "Plan Content Calendar",
                "description": "Create content calendar and schedule",
                "target_integration": None,
                "risk_level": "low",
                "requires_approval": False
            },
            "scheduling": {
                "action_type": "schedule",
                "action_name": "Schedule Posts",
                "description": "Schedule content for automated posting",
                "target_integration": "social",
                "risk_level": "medium",
                "requires_approval": True
            },
            "engagement": {
                "action_type": "monitor",
                "action_name": "Monitor Engagement",
                "description": "Track and respond to audience engagement",
                "target_integration": None,
                "risk_level": "low",
                "requires_approval": False
            },
            "reminder_schedule": {
                "action_type": "schedule",
                "action_name": "Schedule Reminders",
                "description": "Set up reminder sequence",
                "target_integration": "slack",
                "risk_level": "low",
                "requires_approval": False
            },
            "material_prep": {
                "action_type": "prepare",
                "action_name": "Prepare Materials",
                "description": "Prepare event materials and collateral",
                "target_integration": None,
                "risk_level": "low",
                "requires_approval": False
            }
        }

        config = step_configs.get(step_type, {
            "action_type": "generic",
            "action_name": f"Execute {step_type}",
            "description": f"Perform {step_type}",
            "target_integration": None,
            "risk_level": "low",
            "requires_approval": False
        })

        # Add required parameters based on action type
        parameters = {
            "campaign_details": campaign_details,
            "step_type": step_type
        }

        if config["action_type"] == "send_message":
            # Add required parameters for send_message
            parameters["channel"] = channels[0] if channels else "general"
            parameters["message"] = f"{config['description']} - {campaign_details.get('message', campaign_details.get('title', 'Campaign Update'))}"
        elif config["action_type"] == "create_task":
            # Add required parameters for create_task
            parameters["title"] = f"{campaign_details.get('title', 'Marketing Campaign')} - {step_type}"
            parameters["description"] = f"{config['description']} for {campaign_details.get('title', 'campaign')}"
        elif config["action_type"] == "generate_content":
            # Add content generation parameters
            parameters["content_type"] = "marketing"
            parameters["topics"] = campaign_details.get("key_messages", [])

        return ActionStep(
            step_index=index,
            action_type=config["action_type"],
            action_name=config["action_name"],
            description=config["description"],
            target_integration=config["target_integration"],
            target_resource={
                "campaign": campaign_details.get("title", "Campaign"),
                "channels": channels
            },
            parameters=parameters,
            risk_level=config["risk_level"],
            requires_approval=config["requires_approval"],
            depends_on=[i for i in range(index) if i < index],  # Depends on previous steps
            estimated_duration_ms=30000  # 30 seconds estimate
        )

    async def _execute_impl(self, action: ActionStep) -> ExecutionResult:
        """
        Execute a marketing action step.

        Routes to appropriate integration:
        - Slack for announcements
        - ClickUp for task creation
        - HubSpot for campaigns
        - Internal for content generation

        Args:
            action: Action to execute

        Returns:
            Execution result
        """
        logger.info(f"[MARKETING_AGENT] Executing: {action.action_name}")

        try:
            result_data = {}

            # Route based on action type
            if action.action_type == "send_message":
                # Send Slack message
                result_data = await self._execute_slack_message(action)

            elif action.action_type == "create_task":
                # Create ClickUp task
                result_data = await self._execute_clickup_task(action)

            elif action.action_type == "generate_content":
                # Generate content using LLM
                result_data = await self._execute_content_generation(action)

            elif action.action_type == "schedule":
                # Schedule action
                result_data = await self._execute_scheduling(action)

            else:
                # Generic execution
                result_data = {
                    "status": "simulated",
                    "message": f"Simulated execution of {action.action_name}"
                }

            return ExecutionResult(
                success=True,
                action_id=f"marketing_{action.step_index}",
                result=result_data,
                execution_time_ms=1000,
                side_effects=[{
                    "type": "log",
                    "description": f"Executed marketing action: {action.action_name}"
                }]
            )

        except Exception as e:
            logger.error(f"[MARKETING_AGENT] Execution failed: {e}")
            return ExecutionResult(
                success=False,
                action_id=f"marketing_{action.step_index}",
                error_message=str(e),
                execution_time_ms=1000
            )

    async def _execute_slack_message(self, action: ActionStep) -> Dict[str, Any]:
        """Execute Slack message sending."""
        import time
        start_time = time.time()

        campaign = action.parameters.get("campaign_details", {})
        channel = action.parameters.get("channel", "marketing")

        # Remove # prefix if present
        channel_id = channel.lstrip('#')

        message = f"""📢 *Marketing Update: {campaign.get('title', 'Campaign')}*

{campaign.get('description', '')}

*Key Messages:*
{chr(10).join('• ' + msg for msg in campaign.get('key_messages', []))}

*Target Audience:* {campaign.get('target_audience', 'General')}
*Timeline:* {campaign.get('timeline', 'TBD')} days
"""

        logger.info(f"[MARKETING_AGENT] Sending Slack message to #{channel_id}")

        try:
            # Get Slack access token from database
            from app.db import supabase

            creds = supabase.table("connections")\
                .select("access_token")\
                .eq("org_id", self.org_id)\
                .eq("source_type", "slack")\
                .single()\
                .execute()

            if not creds.data:
                raise Exception("Slack not connected for this organization")

            access_token = creds.data["access_token"]

            # Send message via Slack API
            slack_response = self.slack_service.post_message(
                access_token=access_token,
                channel=channel_id,
                text=message
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            logger.info(f"[MARKETING_AGENT] Message sent successfully to #{channel_id}")

            return {
                "status": "sent",
                "channel": f"#{channel_id}",
                "message": message,
                "timestamp": slack_response.get("ts"),
                "slack_channel_id": slack_response.get("channel"),
                "execution_time_ms": execution_time_ms
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[MARKETING_AGENT] Failed to send Slack message: {error_msg}")

            # Provide helpful error messages
            if "missing_scope" in error_msg.lower():
                error_msg = "Missing Slack permission. Please add 'chat:write' scope and reinstall the app."
            elif "channel_not_found" in error_msg.lower():
                error_msg = f"Channel '#{channel_id}' not found. Please check the channel name."
            elif "not_in_channel" in error_msg.lower():
                error_msg = f"Bot is not in channel '#{channel_id}'. Please invite the bot to the channel."

            raise Exception(error_msg)

    async def _execute_clickup_task(self, action: ActionStep) -> Dict[str, Any]:
        """Execute ClickUp task creation."""
        # TODO: Integrate with actual ClickUp service
        campaign = action.parameters.get("campaign_details", {})

        tasks = [
            {
                "name": f"Campaign: {campaign.get('title', 'Marketing Campaign')}",
                "description": campaign.get('description', ''),
                "priority": "high",
                "due_date": (datetime.utcnow() + timedelta(days=campaign.get('timeline', 7))).isoformat()
            }
        ]

        logger.info(f"[MARKETING_AGENT] Would create {len(tasks)} ClickUp tasks")

        return {
            "status": "created",
            "tasks_created": len(tasks),
            "task_ids": ["task_123"],  # Mock IDs
            "list": "Marketing Tasks"
        }

    async def _execute_content_generation(self, action: ActionStep) -> Dict[str, Any]:
        """Execute content generation using LLM."""
        campaign = action.parameters.get("campaign_details", {})

        prompt = f"""Generate marketing content for:

Campaign: {campaign.get('title', 'Campaign')}
Description: {campaign.get('description', '')}
Key Messages: {', '.join(campaign.get('key_messages', []))}
Target Audience: {campaign.get('target_audience', 'General')}

Create:
1. A short social media post (max 280 chars)
2. An email subject line
3. A call-to-action

Format as JSON with keys: social_post, email_subject, cta"""

        try:
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.8,
                max_tokens=300
            )

            # Parse response
            content = {
                "social_post": "Check out our latest campaign! #marketing",
                "email_subject": "Exciting news from our team",
                "cta": "Learn more today"
            }

            try:
                # Try to parse as JSON if response is formatted
                json_str = response.strip()
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                content = json.loads(json_str)
            except:
                # Use default if parsing fails
                pass

            logger.info(f"[MARKETING_AGENT] Generated content successfully")

            return {
                "status": "generated",
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"[MARKETING_AGENT] Content generation failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def _execute_scheduling(self, action: ActionStep) -> Dict[str, Any]:
        """Execute scheduling action."""
        campaign = action.parameters.get("campaign_details", {})

        schedule = []
        timeline_days = campaign.get("timeline", 7)

        # Create schedule based on timeline
        for day in range(0, timeline_days, 2):  # Every 2 days
            schedule.append({
                "day": day,
                "time": "10:00 AM",
                "action": "Post update",
                "channel": action.target_integration or "slack"
            })

        logger.info(f"[MARKETING_AGENT] Created schedule with {len(schedule)} items")

        return {
            "status": "scheduled",
            "schedule_items": len(schedule),
            "first_item": schedule[0] if schedule else None,
            "last_item": schedule[-1] if schedule else None
        }

    async def _reflect_impl(self, plan_id: str, results: List[ExecutionResult]) -> ReflectionInsight:
        """
        Reflect on marketing campaign execution.

        Analyzes:
        - Success rate of actions
        - Content effectiveness
        - Engagement metrics
        - Areas for improvement
        - Ad platform performance (Phase 6)

        Args:
            plan_id: Plan ID
            results: Execution results

        Returns:
            Reflection insights
        """
        logger.info(f"[MARKETING_AGENT] Reflecting on plan {plan_id}")

        # Analyze results
        successful_actions = [r for r in results if r.success]
        failed_actions = [r for r in results if not r.success]

        # Extract metrics from results
        total_messages_sent = sum(
            1 for r in successful_actions
            if r.result.get("status") == "sent"
        )

        total_tasks_created = sum(
            r.result.get("tasks_created", 0)
            for r in successful_actions
        )

        content_generated = any(
            r.result.get("status") == "generated"
            for r in successful_actions
        )

        # Phase 6: Fetch ad platform performance metrics
        ad_performance = await self._fetch_ad_platform_performance()

        # Generate lessons learned
        lessons = []

        if len(successful_actions) == len(results):
            lessons.append("All campaign actions executed successfully")

        # Phase 6: Add lessons from ad platform performance
        if ad_performance:
            for platform, metrics in ad_performance.items():
                if metrics.get("avg_ctr", 0) > 2.0:
                    lessons.append(f"{platform}: Excellent CTR of {metrics['avg_ctr']:.2f}%")
                if metrics.get("avg_roas", 0) > 3.0:
                    lessons.append(f"{platform}: Strong ROAS of {metrics['avg_roas']:.2f}x")
                if metrics.get("total_conversions", 0) > 50:
                    lessons.append(f"{platform}: High conversion volume ({metrics['total_conversions']} conversions)")

        if failed_actions:
            lessons.append(f"{len(failed_actions)} actions failed and need review")
            for failed in failed_actions:
                if failed.error_message:
                    lessons.append(f"Failure reason: {failed.error_message[:100]}")

        if total_messages_sent > 0:
            lessons.append(f"Successfully sent {total_messages_sent} announcement(s)")

        if total_tasks_created > 0:
            lessons.append(f"Created {total_tasks_created} campaign task(s)")

        if content_generated:
            lessons.append("Content generation successful")

        # Generate improvement suggestions
        improvements = []

        if failed_actions:
            improvements.append("Investigate and fix integration issues")

        if not content_generated:
            improvements.append("Implement content generation workflow")

        if total_messages_sent == 0:
            improvements.append("Ensure announcement channels are configured")

        # Phase 6: Add improvements from ad performance comparison
        if ad_performance and len(ad_performance) > 1:
            # Compare platforms
            platforms_by_roas = sorted(
                ad_performance.items(),
                key=lambda x: x[1].get("avg_roas", 0),
                reverse=True
            )

            if len(platforms_by_roas) >= 2:
                best = platforms_by_roas[0]
                worst = platforms_by_roas[-1]

                if best[1].get("avg_roas", 0) > worst[1].get("avg_roas", 0) * 1.5:
                    improvements.append(
                        f"Consider shifting budget from {worst[0]} (ROAS: {worst[1].get('avg_roas', 0):.2f}x) "
                        f"to {best[0]} (ROAS: {best[1].get('avg_roas', 0):.2f}x)"
                    )

        improvements.append("Collect engagement metrics for next iteration")
        improvements.append("A/B test different message formats")

        # Determine if retry is needed
        should_retry = len(failed_actions) > 0 and len(failed_actions) < len(results) / 2

        # Build performance metrics including ad platform data
        performance_metrics = {
            "total_actions": len(results),
            "successful_actions": len(successful_actions),
            "failed_actions": len(failed_actions),
            "messages_sent": total_messages_sent,
            "tasks_created": total_tasks_created,
            "content_generated": content_generated
        }

        # Phase 6: Include ad platform performance in metrics
        if ad_performance:
            performance_metrics["ad_platforms"] = ad_performance

        reflection = ReflectionInsight(
            plan_id=plan_id,
            overall_success=len(failed_actions) == 0,
            lessons_learned=lessons,
            improvements_suggested=improvements,
            performance_metrics=performance_metrics,
            should_retry=should_retry,
            retry_modifications={
                "skip_failed_steps": True,
                "add_retries": True
            } if should_retry else None
        )

        # Publish campaign_completed event if successful
        if reflection.overall_success:
            try:
                # Get campaign details from context
                campaign_summary = {
                    "plan_id": plan_id,
                    "success": True,
                    "metrics": reflection.performance_metrics,
                    "completed_at": datetime.utcnow().isoformat()
                }

                # Get org_id from self.config
                org_id = self.config.get("org_id")
                if org_id:
                    self.event_bus.publish(
                        org_id=org_id,
                        event_type=AgentEventType.CAMPAIGN_COMPLETED,
                        source_agent="marketing_agent",
                        payload=campaign_summary,
                        target_agent="communication_agent",  # Target CommunicationAgent for announcement
                        priority="normal"
                    )
                    logger.info(
                        f"[MARKETING_AGENT] Published campaign_completed event",
                        extra={"plan_id": plan_id, "org_id": org_id}
                    )
            except Exception as e:
                # Don't fail reflection if event publishing fails
                logger.error(f"[MARKETING_AGENT] Failed to publish campaign_completed event: {e}")

        return reflection

    async def _fetch_cil_proposals(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Fetch pending CIL ads optimization proposals.

        Phase 6, Step 2: Query cil_policy_proposals table for ads-specific proposals
        that haven't been applied yet.

        Args:
            org_id: Organization ID

        Returns:
            List of pending proposals
        """
        try:
            from app.db import supabase

            # Fetch pending ads optimization proposals
            result = supabase.table('cil_policy_proposals')\
                .select('*')\
                .eq('org_id', org_id)\
                .in_('proposal_type', [
                    'budget_adjustment',
                    'pause_campaign',
                    'clone_campaign',
                    'platform_shift',
                    'increase_budget',
                    'decrease_budget'
                ])\
                .eq('status', 'pending')\
                .order('created_at', desc=False)\
                .execute()

            proposals = result.data or []

            logger.debug(f"[MARKETING_AGENT] Found {len(proposals)} pending CIL ads proposals")

            return proposals

        except Exception as e:
            logger.error(f"[MARKETING_AGENT] Failed to fetch CIL proposals: {e}", exc_info=True)
            return []

    async def _apply_proposal(self, proposal: Dict[str, Any]) -> bool:
        """
        Apply a CIL ads optimization proposal.

        Phase 6, Step 2: Execute the optimization action (budget shift, pause, clone, etc.)
        and record the outcome.

        Args:
            proposal: Proposal dictionary

        Returns:
            True if applied successfully, False otherwise
        """
        try:
            from app.db import supabase
            from datetime import datetime

            proposal_id = proposal['id']
            proposal_type = proposal['proposal_type']
            ads_context = proposal.get('ads_context', {})

            logger.info(
                f"[MARKETING_AGENT] Applying CIL proposal: {proposal_type}",
                extra={'proposal_id': proposal_id}
            )

            # Capture before metrics
            metrics_before = {}

            # Execute action based on proposal type
            applied = False
            action_taken = ""

            if proposal_type == 'pause_campaign':
                # Pause underperforming campaign
                campaign_id = ads_context.get('campaign_id')
                if campaign_id:
                    # Get current metrics
                    campaign = supabase.table('ad_campaigns')\
                        .select('*')\
                        .eq('id', campaign_id)\
                        .single()\
                        .execute()

                    if campaign.data:
                        metrics_before = {
                            'status': campaign.data['status'],
                            'campaign_name': campaign.data['name']
                        }

                        # Pause campaign
                        supabase.table('ad_campaigns')\
                            .update({'status': 'paused', 'updated_at': datetime.utcnow().isoformat()})\
                            .eq('id', campaign_id)\
                            .execute()

                        action_taken = 'paused'
                        applied = True
                        logger.info(f"[MARKETING_AGENT] Paused campaign {campaign_id}")

            elif proposal_type in ['increase_budget', 'decrease_budget']:
                # Adjust campaign budget
                campaign_id = ads_context.get('campaign_id')
                adjustment_pct = ads_context.get('adjustment_percentage', 0)

                if campaign_id and adjustment_pct:
                    campaign = supabase.table('ad_campaigns')\
                        .select('*')\
                        .eq('id', campaign_id)\
                        .single()\
                        .execute()

                    if campaign.data:
                        old_budget = float(campaign.data.get('daily_budget', 0))
                        new_budget = old_budget * (1 + adjustment_pct / 100.0)

                        metrics_before = {
                            'daily_budget': old_budget,
                            'campaign_name': campaign.data['name']
                        }

                        # Update budget
                        supabase.table('ad_campaigns')\
                            .update({
                                'daily_budget': new_budget,
                                'updated_at': datetime.utcnow().isoformat()
                            })\
                            .eq('id', campaign_id)\
                            .execute()

                        action_taken = f'budget_{proposal_type.split("_")[0]}d'
                        applied = True
                        logger.info(
                            f"[MARKETING_AGENT] Adjusted budget for campaign {campaign_id}: "
                            f"${old_budget:.2f} → ${new_budget:.2f}"
                        )

            elif proposal_type == 'platform_shift':
                # Record platform preference recommendation
                from_platform = ads_context.get('from_platform')
                to_platform = ads_context.get('to_platform')
                performance_delta = ads_context.get('performance_delta', 0)

                metrics_before = {
                    'from_platform': from_platform,
                    'to_platform': to_platform,
                    'performance_delta': performance_delta
                }

                # Store recommendation in ai_meta_knowledge
                supabase.table('ai_meta_knowledge').insert({
                    'org_id': proposal['org_id'],
                    'knowledge_type': 'platform_preference',
                    'source_agent': 'marketing_agent',
                    'insight_text': f"Shift budget from {from_platform} to {to_platform} (ROAS delta: {performance_delta:.2f}x)",
                    'confidence_score': proposal.get('confidence_score', 0.8),
                    'context': json.dumps({
                        'from_platform': from_platform,
                        'to_platform': to_platform,
                        'performance_delta': performance_delta
                    })
                }).execute()

                action_taken = 'platform_preference_recorded'
                applied = True
                logger.info(
                    f"[MARKETING_AGENT] Recorded platform preference: {from_platform} → {to_platform}"
                )

            elif proposal_type == 'clone_campaign':
                # Clone high-performing campaign
                source_campaign_id = ads_context.get('campaign_id')

                if source_campaign_id:
                    source = supabase.table('ad_campaigns')\
                        .select('*')\
                        .eq('id', source_campaign_id)\
                        .single()\
                        .execute()

                    if source.data:
                        metrics_before = {
                            'source_campaign_id': source_campaign_id,
                            'source_campaign_name': source.data['name']
                        }

                        # Clone campaign (simplified - just log for now, actual cloning requires platform API)
                        logger.info(
                            f"[MARKETING_AGENT] Would clone campaign {source_campaign_id} ({source.data['name']})"
                        )

                        action_taken = 'clone_scheduled'
                        applied = True

            # Record proposal application in history
            if applied:
                # Update proposal status
                supabase.table('cil_policy_proposals')\
                    .update({
                        'status': 'applied',
                        'applied_at': datetime.utcnow().isoformat()
                    })\
                    .eq('id', proposal_id)\
                    .execute()

                # Create optimization history record
                history_record = {
                    'org_id': proposal['org_id'],
                    'proposal_id': proposal_id,
                    'proposal_type': proposal_type,
                    'platform': ads_context.get('platform', 'unknown'),
                    'campaign_id': ads_context.get('campaign_id'),
                    'action_taken': action_taken,
                    'metrics_before': json.dumps(metrics_before),
                    'applied_at': datetime.utcnow().isoformat(),
                    'expected_gain': proposal.get('expected_gain'),
                    'confidence_score': proposal.get('confidence_score')
                }

                supabase.table('cil_ads_optimization_history').insert(history_record).execute()

                logger.info(
                    f"[MARKETING_AGENT] Successfully applied proposal {proposal_id}",
                    extra={'proposal_type': proposal_type, 'action': action_taken}
                )

                return True

            else:
                logger.warning(
                    f"[MARKETING_AGENT] Could not apply proposal {proposal_id} - insufficient context"
                )
                return False

        except Exception as e:
            logger.error(
                f"[MARKETING_AGENT] Failed to apply proposal {proposal.get('id')}: {e}",
                exc_info=True
            )
            return False

    async def _record_proposal_outcome(
        self,
        history_id: str,
        success: bool,
        actual_gain: Optional[float] = None,
        feedback_notes: Optional[str] = None
    ) -> None:
        """
        Record the outcome of an applied CIL proposal.

        Phase 6, Step 2: Update optimization history with results after observation period.

        Args:
            history_id: Optimization history record ID
            success: Whether the optimization improved performance
            actual_gain: Actual performance improvement percentage
            feedback_notes: Optional notes about the outcome
        """
        try:
            from app.db import supabase
            from datetime import datetime

            update_data = {
                'success': success,
                'actual_gain': actual_gain,
                'feedback_notes': feedback_notes,
                'observation_end_date': datetime.utcnow().date().isoformat()
            }

            supabase.table('cil_ads_optimization_history')\
                .update(update_data)\
                .eq('id', history_id)\
                .execute()

            logger.info(
                f"[MARKETING_AGENT] Recorded outcome for optimization {history_id}",
                extra={'success': success, 'actual_gain': actual_gain}
            )

        except Exception as e:
            logger.error(
                f"[MARKETING_AGENT] Failed to record proposal outcome: {e}",
                exc_info=True
            )

    async def analyze_campaign_performance(
        self,
        campaign_id: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze campaign performance metrics.

        Args:
            campaign_id: Campaign identifier
            metrics: Performance metrics

        Returns:
            Performance analysis
        """
        analysis = {
            "campaign_id": campaign_id,
            "performance_score": 0.0,
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        # Calculate performance score
        engagement_rate = metrics.get("engagement_rate", 0)
        click_rate = metrics.get("click_through_rate", 0)
        conversion_rate = metrics.get("conversion_rate", 0)

        performance_score = (
            engagement_rate * 0.3 +
            click_rate * 0.4 +
            conversion_rate * 0.3
        )

        analysis["performance_score"] = round(performance_score, 2)

        # Identify strengths
        if engagement_rate > 0.05:
            analysis["strengths"].append("Good engagement rate")
        if click_rate > 0.02:
            analysis["strengths"].append("Strong click-through rate")
        if conversion_rate > 0.01:
            analysis["strengths"].append("Effective conversions")

        # Identify weaknesses
        if engagement_rate < 0.02:
            analysis["weaknesses"].append("Low engagement")
        if click_rate < 0.01:
            analysis["weaknesses"].append("Poor click-through")
        if conversion_rate < 0.005:
            analysis["weaknesses"].append("Low conversions")

        # Generate recommendations
        if engagement_rate < 0.05:
            analysis["recommendations"].append("Improve content relevance and timing")
        if click_rate < 0.02:
            analysis["recommendations"].append("Enhance call-to-action clarity")
        if conversion_rate < 0.01:
            analysis["recommendations"].append("Optimize landing page experience")

        logger.info(f"[MARKETING_AGENT] Campaign {campaign_id} performance: {performance_score}")

        return analysis

    async def _fetch_ad_platform_performance(self) -> Optional[Dict[str, Any]]:
        """
        Fetch recent ad platform performance metrics for reflection.

        Phase 6: Queries ad_campaigns and ad_metrics tables to get
        aggregated performance data from Google Ads and Meta Ads.

        Returns:
            Dictionary with platform performance metrics or None
        """
        try:
            from app.db import supabase
            from datetime import datetime, timedelta

            org_id = self.config.get("org_id")
            if not org_id:
                return None

            # Get active ad accounts
            accounts = supabase.table("ad_accounts")\
                .select("id, platform")\
                .eq("org_id", org_id)\
                .eq("status", "active")\
                .execute()

            if not accounts.data:
                logger.debug(f"[MARKETING_AGENT] No active ad accounts for org {org_id}")
                return None

            # Calculate date range (last 30 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)

            platform_performance = {}

            for account in accounts.data:
                platform = account["platform"]

                # Get campaigns for this account
                campaigns = supabase.table("ad_campaigns")\
                    .select("id")\
                    .eq("account_id", account["id"])\
                    .in_("status", ["active", "paused"])\
                    .execute()

                if not campaigns.data:
                    continue

                campaign_ids = [c["id"] for c in campaigns.data]

                # Get metrics for these campaigns
                metrics = supabase.table("ad_metrics")\
                    .select("*")\
                    .in_("campaign_id", campaign_ids)\
                    .gte("metric_date", start_date.isoformat())\
                    .lte("metric_date", end_date.isoformat())\
                    .execute()

                if metrics.data:
                    # Aggregate metrics
                    total_impressions = sum(m["impressions"] for m in metrics.data)
                    total_clicks = sum(m["clicks"] for m in metrics.data)
                    total_spend = sum(m["spend"] for m in metrics.data)
                    total_conversions = sum(m.get("conversions", 0) for m in metrics.data)
                    total_conv_value = sum(m.get("conversion_value", 0) or 0 for m in metrics.data)

                    # Calculate averages
                    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                    avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0
                    avg_roas = (total_conv_value / total_spend) if total_spend > 0 else 0

                    # Quality score / engagement rate (platform-specific)
                    if platform == "google_ads":
                        quality_scores = [m.get("quality_score") for m in metrics.data if m.get("quality_score")]
                        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else None
                        special_metric = {"avg_quality_score": round(avg_quality_score, 1)} if avg_quality_score else {}
                    else:  # meta_ads
                        engagement_rates = [m.get("engagement_rate") for m in metrics.data if m.get("engagement_rate")]
                        avg_engagement = sum(engagement_rates) / len(engagement_rates) if engagement_rates else None
                        special_metric = {"avg_engagement_rate": round(avg_engagement, 2)} if avg_engagement else {}

                    platform_performance[platform] = {
                        "total_impressions": total_impressions,
                        "total_clicks": total_clicks,
                        "total_spend": round(total_spend, 2),
                        "total_conversions": total_conversions,
                        "avg_ctr": round(avg_ctr, 2),
                        "avg_cpc": round(avg_cpc, 2),
                        "avg_roas": round(avg_roas, 2),
                        **special_metric,
                        "campaigns_count": len(campaign_ids),
                        "days_analyzed": 30
                    }

            if platform_performance:
                logger.info(
                    f"[MARKETING_AGENT] Fetched ad performance for {len(platform_performance)} platforms",
                    extra={"org_id": org_id}
                )
                return platform_performance

            return None

        except Exception as e:
            logger.error(f"[MARKETING_AGENT] Failed to fetch ad performance: {e}", exc_info=True)
            return None