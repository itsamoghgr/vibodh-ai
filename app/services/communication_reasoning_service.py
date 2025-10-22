"""
Communication Reasoning Service - Phase 4 Enhanced
Intelligent reasoning layer for communication decisions

Provides:
- Intent classification (informational/urgent/strategic)
- Module recommendation (which AI modules to activate)
- Appropriateness verification (pre-execution checks)
- Contextual analysis (timing, audience, sensitivity)
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from supabase import Client
from app.core.config import settings
from app.core.logging import logger
import requests
import json


class CommunicationType(str, Enum):
    """Types of communication intents"""
    INFORMATIONAL = "informational"  # General updates, FYI messages
    URGENT = "urgent"  # Time-sensitive, requires immediate attention
    STRATEGIC = "strategic"  # Important decisions, announcements
    ROUTINE = "routine"  # Regular check-ins, status updates


class CommunicationIntent(BaseModel):
    """Structured communication intent analysis"""
    intent_type: CommunicationType
    urgency_score: float  # 0.0-1.0
    recommended_modules: List[str]  # ["rag", "kg", "memory", "insight"]
    suggested_audience: List[str]  # ["#channel", "person@email"]
    reasoning: str  # Explanation of classification
    confidence: float  # 0.0-1.0


class VerificationResult(BaseModel):
    """Pre-execution verification result"""
    appropriate: bool
    confidence: float  # 0.0-1.0
    reason: str
    suggested_changes: Optional[List[str]] = None


class CommunicationReasoningService:
    """
    Reasoning service for intelligent communication decisions.

    Acts as the "brain" that thinks before the agent acts.
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.groq_api_key = settings.GROQ_API_KEY

    async def analyze_communication_request(
        self,
        query: str,
        org_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> CommunicationIntent:
        """
        Analyze communication request to determine intent and strategy.

        Args:
            query: User's communication request
            org_id: Organization ID
            user_id: Optional user ID
            context: Additional context

        Returns:
            CommunicationIntent with classification and recommendations
        """
        logger.info(f"[COMM_REASONING] Analyzing: {query[:80]}...")

        try:
            # Step 1: Classify intent type
            intent_type, urgency_score = await self._classify_intent(query)

            # Step 2: Recommend modules based on intent
            recommended_modules = self._recommend_modules(query, intent_type)

            # Step 3: Identify suggested audience
            suggested_audience = await self._identify_audience(query, org_id)

            # Step 4: Generate reasoning explanation
            reasoning = self._generate_reasoning(query, intent_type, urgency_score)

            result = CommunicationIntent(
                intent_type=intent_type,
                urgency_score=urgency_score,
                recommended_modules=recommended_modules,
                suggested_audience=suggested_audience,
                reasoning=reasoning,
                confidence=0.85  # Could be calculated from LLM confidence
            )

            logger.info(
                f"[COMM_REASONING] Classified as {intent_type.value} "
                f"(urgency: {urgency_score:.2f})"
            )

            return result

        except Exception as e:
            logger.error(f"[COMM_REASONING] Analysis failed: {e}")
            # Return safe default
            return CommunicationIntent(
                intent_type=CommunicationType.INFORMATIONAL,
                urgency_score=0.5,
                recommended_modules=["rag"],
                suggested_audience=[],
                reasoning="Fallback classification due to analysis error",
                confidence=0.3
            )

    async def _classify_intent(self, query: str) -> Tuple[CommunicationType, float]:
        """
        Classify the communication intent using LLM.

        Returns:
            Tuple of (intent_type, urgency_score)
        """
        # Keyword-based quick classification (fast path)
        query_lower = query.lower()

        # Urgent indicators
        urgent_keywords = ["urgent", "asap", "immediately", "critical", "emergency", "now"]
        if any(keyword in query_lower for keyword in urgent_keywords):
            return CommunicationType.URGENT, 0.9

        # Strategic indicators
        strategic_keywords = [
            "announce", "launch", "decision", "strategy", "important",
            "everyone", "all teams", "organization"
        ]
        if any(keyword in query_lower for keyword in strategic_keywords):
            return CommunicationType.STRATEGIC, 0.7

        # Routine indicators
        routine_keywords = ["update", "status", "check-in", "reminder", "fyi"]
        if any(keyword in query_lower for keyword in routine_keywords):
            return CommunicationType.ROUTINE, 0.3

        # Default to informational with LLM refinement
        return await self._llm_classify_intent(query)

    async def _llm_classify_intent(self, query: str) -> Tuple[CommunicationType, float]:
        """
        Use LLM for nuanced intent classification.
        """
        try:
            prompt = f"""Classify this communication request:

Query: "{query}"

Classify as one of:
- URGENT: Time-sensitive, requires immediate attention (urgency: 0.8-1.0)
- STRATEGIC: Important announcement or decision (urgency: 0.6-0.8)
- INFORMATIONAL: General update or FYI (urgency: 0.3-0.6)
- ROUTINE: Regular status update (urgency: 0.0-0.3)

Return ONLY a JSON object:
{{
    "type": "URGENT|STRATEGIC|INFORMATIONAL|ROUTINE",
    "urgency": 0.0-1.0
}}"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": settings.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 100
                },
                timeout=10
            )

            if response.status_code == 200:
                result_text = response.json()["choices"][0]["message"]["content"].strip()
                # Extract JSON from response
                result_json = json.loads(result_text)

                intent_type = CommunicationType(result_json["type"].lower())
                urgency = float(result_json["urgency"])

                return intent_type, urgency

            logger.warning(f"[COMM_REASONING] LLM classification failed, using default")
            return CommunicationType.INFORMATIONAL, 0.5

        except Exception as e:
            logger.error(f"[COMM_REASONING] LLM classification error: {e}")
            return CommunicationType.INFORMATIONAL, 0.5

    def _recommend_modules(self, query: str, intent_type: CommunicationType) -> List[str]:
        """
        Recommend which AI modules should be activated.

        Returns:
            List of module names: ["rag", "kg", "memory", "insight"]
        """
        modules = ["rag"]  # Always use RAG for context

        query_lower = query.lower()

        # Use Knowledge Graph for relationship/entity queries
        if any(word in query_lower for word in ["team", "person", "who", "channel", "group"]):
            modules.append("kg")

        # Use Memory for historical context
        if any(word in query_lower for word in ["recent", "last", "previous", "history", "past"]):
            modules.append("memory")

        # Use Insights for strategic communications
        if intent_type == CommunicationType.STRATEGIC:
            modules.append("insight")
            modules.append("kg")  # Strategic needs org structure

        # Urgent communications need full context
        if intent_type == CommunicationType.URGENT:
            modules.extend(["kg", "memory", "insight"])

        return list(set(modules))  # Remove duplicates

    async def _identify_audience(self, query: str, org_id: str) -> List[str]:
        """
        Identify suggested audience using LLM to extract all communication details.

        Uses intelligent LLM extraction instead of fragile regex patterns.
        Returns channels, people, and teams mentioned in the query.
        """
        try:
            prompt = f"""Analyze this communication request and extract ALL relevant information.

Query: "{query}"

Extract the following information:
1. **Channels**: Any Slack channels mentioned (e.g., #engineering, private-ch, general)
2. **People**: Any specific people mentioned by name
3. **Teams**: Any teams or groups mentioned (e.g., "engineering team", "everyone", "all")
4. **Message Content**: The actual message to send or topic to communicate
5. **Additional Details**: Dates, times, or other important context

Return ONLY a JSON object with this structure:

{{
  "channels": [
    {{"name": "private-ch", "confidence": 0.95, "mentioned_as": "private-ch channel"}},
    {{"name": "engineering", "confidence": 0.9, "mentioned_as": "#engineering"}}
  ],
  "people": [
    {{"name": "John Doe", "role": "mentioned recipient"}}
  ],
  "teams": [
    {{"name": "everyone", "scope": "organization-wide"}}
  ],
  "message_content": "The actual message or topic to communicate",
  "additional_details": {{
    "dates": ["10th Nov 2025"],
    "urgency": "normal",
    "context": "launch announcement"
  }}
}}

IMPORTANT:
- Channel names should NOT include # symbol in the "name" field
- If "private-ch channel" is mentioned, extract as "private-ch"
- If "#engineering" is mentioned, extract as "engineering"
- If no channels/people/teams mentioned, use empty arrays
- Be generous with confidence scores (0.8-1.0 for clear mentions)

Examples:

Query: "send hello to #engineering"
Response:
{{
  "channels": [{{"name": "engineering", "confidence": 1.0, "mentioned_as": "#engineering"}}],
  "people": [],
  "teams": [],
  "message_content": "hello",
  "additional_details": {{"urgency": "normal"}}
}}

Query: "inform everyone about the launch date - 10th Nov 2025, and in private-ch channel"
Response:
{{
  "channels": [{{"name": "private-ch", "confidence": 1.0, "mentioned_as": "private-ch channel"}}],
  "people": [],
  "teams": [{{"name": "everyone", "scope": "organization-wide"}}],
  "message_content": "launch date - 10th Nov 2025",
  "additional_details": {{"dates": ["10th Nov 2025"], "context": "launch announcement"}}
}}

Now extract from this query:"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": settings.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"}
                },
                timeout=10
            )

            if response.status_code == 200:
                result = json.loads(response.json()["choices"][0]["message"]["content"])

                # Store full extraction result for later use
                self._last_extraction = result

                # Extract channel names and format with #
                suggested = []
                for channel in result.get("channels", []):
                    channel_name = channel.get("name", "")
                    if channel_name:
                        # Add # prefix if not present
                        formatted_channel = f"#{channel_name}" if not channel_name.startswith("#") else channel_name
                        suggested.append(formatted_channel)
                        logger.info(
                            f"[COMM_REASONING] Extracted channel: {formatted_channel} "
                            f"(confidence: {channel.get('confidence', 0):.2f}, "
                            f"mentioned as: '{channel.get('mentioned_as', '')}')"
                        )

                # Also add people and teams to suggested audience
                for person in result.get("people", []):
                    suggested.append(person.get("name", ""))

                for team in result.get("teams", []):
                    team_name = team.get("name", "")
                    if team_name and team_name not in suggested:
                        suggested.append(team_name)

                logger.info(f"[COMM_REASONING] Full extraction: {json.dumps(result, indent=2)}")

                return suggested

            logger.warning("[COMM_REASONING] LLM extraction failed, returning empty list")
            return []

        except Exception as e:
            logger.error(f"[COMM_REASONING] LLM audience extraction error: {e}")
            return []

    def _generate_reasoning(
        self,
        query: str,
        intent_type: CommunicationType,
        urgency_score: float
    ) -> str:
        """Generate human-readable reasoning explanation."""

        reasoning_parts = [
            f"Classified as {intent_type.value}",
            f"urgency level {urgency_score:.2f}"
        ]

        if urgency_score > 0.8:
            reasoning_parts.append("requires immediate attention")
        elif urgency_score > 0.6:
            reasoning_parts.append("moderately time-sensitive")
        else:
            reasoning_parts.append("can be scheduled appropriately")

        return ", ".join(reasoning_parts)

    async def verify_appropriateness(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> VerificationResult:
        """
        Pre-execution verification to ensure communication is appropriate.

        Checks:
        - Timing (business hours, weekends)
        - Frequency (not spamming)
        - Tone (appropriate for audience)
        - Sensitivity (no confidential info in public channels)

        Args:
            message: Message to send
            context: Context including channel, time, recent activity

        Returns:
            VerificationResult with appropriateness decision
        """
        logger.info(f"[COMM_REASONING] Verifying appropriateness...")

        # Check 1: Timing appropriateness
        current_hour = datetime.now().hour
        is_business_hours = 9 <= current_hour <= 17
        is_weekend = datetime.now().weekday() >= 5

        timing_issues = []
        if not is_business_hours and "urgent" in message.lower():
            timing_issues.append("Urgent message outside business hours may not be seen")

        if is_weekend and "important" in message.lower():
            timing_issues.append("Important message on weekend may be delayed")

        # Check 2: Message frequency (avoid spam)
        channel = context.get('channel', '')
        recent_messages = context.get('recent_messages', [])

        if len(recent_messages) > 5:
            timing_issues.append("High message frequency in channel recently")

        # Check 3: Use LLM for nuanced verification
        llm_verification = await self._llm_verify_appropriateness(message, context)

        # Combine checks
        if timing_issues:
            logger.warning(f"[COMM_REASONING] Timing issues: {timing_issues}")

        # If critical issues found, mark as inappropriate
        if len(timing_issues) >= 2 or not llm_verification['appropriate']:
            return VerificationResult(
                appropriate=False,
                confidence=0.7,
                reason="; ".join(timing_issues + [llm_verification.get('reason', '')]),
                suggested_changes=[
                    "Consider scheduling for business hours",
                    "Check if immediate delivery is necessary"
                ]
            )

        # Passed verification
        return VerificationResult(
            appropriate=True,
            confidence=0.9 if not timing_issues else 0.75,
            reason="Message is contextually appropriate",
            suggested_changes=[]
        )

    async def _llm_verify_appropriateness(
        self,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM to verify message appropriateness with context awareness.
        """
        try:
            channel = context.get('channel', 'unknown')
            time_of_day = datetime.now().strftime("%I:%M %p")
            day_of_week = datetime.now().strftime("%A")

            prompt = f"""Verify if this communication is appropriate:

Message: "{message}"
Channel: {channel}
Time: {day_of_week} {time_of_day}

Consider:
- Is the timing appropriate?
- Is the tone suitable for the audience?
- Does the message contain sensitive information?
- Is it contextually relevant?

Return ONLY a JSON object:
{{
    "appropriate": true|false,
    "confidence": 0.0-1.0,
    "reason": "brief explanation"
}}"""

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": settings.GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": 150
                },
                timeout=10
            )

            if response.status_code == 200:
                result_text = response.json()["choices"][0]["message"]["content"].strip()
                return json.loads(result_text)

            # Default to appropriate if LLM fails
            return {"appropriate": True, "confidence": 0.5, "reason": "LLM verification unavailable"}

        except Exception as e:
            logger.error(f"[COMM_REASONING] LLM verification error: {e}")
            return {"appropriate": True, "confidence": 0.5, "reason": "Verification check failed"}


def get_communication_reasoning_service(supabase: Client) -> CommunicationReasoningService:
    """Factory function to get CommunicationReasoningService instance."""
    return CommunicationReasoningService(supabase)
