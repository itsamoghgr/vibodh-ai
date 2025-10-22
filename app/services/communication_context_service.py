"""
Communication Context Service - Phase 4 Enhanced
Gathers rich context from KG and Memory for intelligent communications

Provides:
- Audience identification (who should receive messages)
- Channel discovery (preferred communication channels)
- Historical context (recent relevant information)
- Relationship mapping (team structure, reporting lines)
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel
from supabase import Client
from app.services.kg_service import get_kg_service
from app.services.memory_service import get_memory_service
from app.core.logging import logger


class AudienceInfo(BaseModel):
    """Information about communication audience"""
    channels: List[str]  # ["#marketing", "#general"]
    people: List[Dict[str, str]]  # [{"name": "Sarah", "role": "Marketing Lead"}]
    teams: List[str]  # ["Marketing Team", "Sales Team"]
    confidence: float  # 0.0-1.0


class CommunicationContext(BaseModel):
    """Rich context for communication planning"""
    topic: str  # Extracted topic (e.g., "marketing campaign")
    audience: AudienceInfo
    recent_context: List[Dict[str, Any]]  # Recent relevant memories
    related_entities: List[Dict[str, Any]]  # Related KG entities
    suggested_timing: Optional[str] = None  # e.g., "afternoon preferred"
    communication_patterns: Dict[str, Any] = {}  # Historical patterns


class CommunicationContextService:
    """
    Service for gathering rich contextual information for communications.

    Integrates with Knowledge Graph and Memory to understand:
    - Who should be notified
    - Which channels to use
    - What context is relevant
    - When to communicate
    """

    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.kg_service = get_kg_service(supabase)
        self.memory_service = get_memory_service(supabase)

    async def gather_context(
        self,
        query: str,
        org_id: str,
        intent_type: str = "informational",
        suggested_audience: Optional[List[str]] = None
    ) -> CommunicationContext:
        """
        Gather comprehensive context for communication planning.

        Args:
            query: User's communication request
            org_id: Organization ID
            intent_type: Type of communication intent
            suggested_audience: Explicitly mentioned channels/people from query

        Returns:
            CommunicationContext with all relevant information
        """
        logger.info(f"[COMM_CONTEXT] Gathering context for: {query[:80]}...")

        try:
            # Step 1: Extract topic from query
            topic = self._extract_topic(query)

            # Step 2: Identify audience using KG, prioritizing explicit mentions
            audience = await self._identify_audience_from_kg(topic, org_id, suggested_audience)

            # Step 3: Gather recent relevant context from Memory
            recent_context = await self._gather_memory_context(topic, org_id)

            # Step 4: Find related entities from KG
            related_entities = await self._get_related_entities(topic, org_id)

            # Step 5: Analyze communication patterns
            patterns = await self._analyze_communication_patterns(org_id, topic)

            # Step 6: Suggest optimal timing
            suggested_timing = self._suggest_timing(patterns, intent_type)

            context = CommunicationContext(
                topic=topic,
                audience=audience,
                recent_context=recent_context,
                related_entities=related_entities,
                suggested_timing=suggested_timing,
                communication_patterns=patterns
            )

            logger.info(
                f"[COMM_CONTEXT] Context gathered: "
                f"topic='{topic}', "
                f"channels={len(audience.channels)}, "
                f"memories={len(recent_context)}"
            )

            return context

        except Exception as e:
            logger.error(f"[COMM_CONTEXT] Context gathering failed: {e}")
            # Return minimal safe context
            return CommunicationContext(
                topic="general",
                audience=AudienceInfo(
                    channels=["#general"],
                    people=[],
                    teams=[],
                    confidence=0.3
                ),
                recent_context=[],
                related_entities=[],
                suggested_timing=None,
                communication_patterns={}
            )

    def _extract_topic(self, query: str) -> str:
        """
        Extract main topic from communication query.

        Examples:
        - "notify marketing about campaign" → "campaign"
        - "tell the team about Q4 results" → "Q4 results"
        """
        import re

        # Common topic patterns
        patterns = [
            r'about\s+(?:the\s+)?([a-zA-Z0-9\s]+?)(?:\s+to|\s+in|$)',  # "about X"
            r'regarding\s+(?:the\s+)?([a-zA-Z0-9\s]+?)(?:\s+to|\s+in|$)',  # "regarding X"
            r'(?:of|for)\s+(?:the\s+)?([a-zA-Z0-9\s]+?)(?:\s+to|\s+in|$)',  # "for X"
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if len(topic) > 3:  # Avoid single words like "the", "and"
                    return topic

        # Fallback: Extract key nouns using simple heuristics
        words = query.lower().split()
        key_nouns = [
            "campaign", "project", "launch", "results", "update",
            "meeting", "deadline", "release", "product", "feature"
        ]

        for noun in key_nouns:
            if noun in words:
                return noun

        return "general update"

    async def _identify_audience_from_kg(
        self,
        topic: str,
        org_id: str,
        suggested_audience: Optional[List[str]] = None
    ) -> AudienceInfo:
        """
        Identify communication audience using Knowledge Graph.

        Finds:
        - Relevant teams based on topic
        - Appropriate channels
        - Key people to notify

        Args:
            topic: Extracted topic from query
            org_id: Organization ID
            suggested_audience: Explicitly mentioned channels/people (takes priority)
        """
        try:
            channels = []
            people = []
            teams = []

            # PRIORITY 1: Use explicitly mentioned channels from user's query
            if suggested_audience:
                for item in suggested_audience:
                    if item.startswith('#'):
                        # It's a channel
                        if item not in channels:
                            channels.append(item)
                            logger.info(f"[COMM_CONTEXT] Using explicitly mentioned channel: {item}")
                    elif '@' in item:
                        # It's an email/person
                        people.append({"name": item, "role": "Mentioned"})

            # PRIORITY 2: If no explicit channels, try topic-based matching
            if not channels:
                channels = self._match_topic_to_channels(topic)

            # Calculate confidence - higher if explicit mentions
            if suggested_audience and channels:
                confidence = 0.95  # High confidence when user explicitly specified
            elif channels and (people or teams):
                confidence = 0.9
            else:
                confidence = 0.6

            return AudienceInfo(
                channels=channels or ["#general"],
                people=people,
                teams=teams,
                confidence=confidence
            )

        except Exception as e:
            logger.error(f"[COMM_CONTEXT] Audience identification failed: {e}")
            return AudienceInfo(
                channels=["#general"],
                people=[],
                teams=[],
                confidence=0.3
            )

    def _match_topic_to_channels(self, topic: str) -> List[str]:
        """
        Match topic to common channel names using heuristics.
        """
        topic_lower = topic.lower()

        # Common topic → channel mappings
        mappings = {
            "marketing": ["#marketing", "#marketing-updates"],
            "campaign": ["#marketing", "#campaigns"],
            "sales": ["#sales", "#sales-team"],
            "engineering": ["#engineering", "#dev"],
            "product": ["#product", "#product-updates"],
            "design": ["#design", "#design-team"],
            "launch": ["#general", "#announcements"],
            "release": ["#releases", "#announcements"],
            "meeting": ["#general", "#meetings"],
            "update": ["#general", "#updates"],
        }

        for keyword, channels in mappings.items():
            if keyword in topic_lower:
                return channels

        return []

    async def _gather_memory_context(
        self,
        topic: str,
        org_id: str,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Gather recent relevant memories about the topic.

        Returns:
            List of memory entries with relevant context
        """
        try:
            # Calculate date threshold
            since_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()

            # Search memories using topic keywords
            memories = self.memory_service.search_memories(
                org_id=org_id,
                query=topic,
                min_importance=0.3,
                limit=5
            )

            # Filter to recent memories
            recent_memories = [
                mem for mem in memories
                if mem.get('created_at', '') >= since_date
            ]

            # Format for context
            context_items = []
            for mem in recent_memories:
                context_items.append({
                    "title": mem.get('title', 'Untitled'),
                    "content": mem.get('content', '')[:200],  # First 200 chars
                    "date": mem.get('created_at', ''),
                    "importance": mem.get('importance', 0.5)
                })

            logger.info(f"[COMM_CONTEXT] Found {len(context_items)} relevant memories")
            return context_items

        except Exception as e:
            logger.error(f"[COMM_CONTEXT] Memory context gathering failed: {e}")
            return []

    async def _get_related_entities(
        self,
        topic: str,
        org_id: str
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to the topic from Knowledge Graph.
        """
        try:
            # Search for all entity types related to topic
            entities = self.kg_service.search_entities_by_name(
                org_id=org_id,
                query=topic,
                limit=10
            )

            # Format entity information
            related = []
            for entity in entities:
                related.append({
                    "name": entity.get('name', 'Unknown'),
                    "type": entity.get('type', 'unknown'),
                    "metadata": entity.get('metadata', {})
                })

            return related

        except Exception as e:
            logger.error(f"[COMM_CONTEXT] Related entities fetch failed: {e}")
            return []

    async def _analyze_communication_patterns(
        self,
        org_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Analyze historical communication patterns.

        Returns patterns like:
        - Best time of day for engagement
        - Preferred channels
        - Response rates by hour
        """
        try:
            # Query recent message events
            result = self.supabase.table("events")\
                .select("payload, happened_at")\
                .eq("org_id", org_id)\
                .eq("source", "slack")\
                .order("happened_at", desc=True)\
                .limit(100)\
                .execute()

            if not result.data:
                return {}

            # Analyze timing patterns
            hour_counts = {}
            for event in result.data:
                happened_at = datetime.fromisoformat(event['happened_at'])
                hour = happened_at.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

            # Find peak engagement hour
            if hour_counts:
                peak_hour = max(hour_counts, key=hour_counts.get)
                patterns = {
                    "peak_hour": peak_hour,
                    "activity_by_hour": hour_counts,
                    "total_messages": len(result.data)
                }
            else:
                patterns = {}

            return patterns

        except Exception as e:
            logger.error(f"[COMM_CONTEXT] Pattern analysis failed: {e}")
            return {}

    def _suggest_timing(
        self,
        patterns: Dict[str, Any],
        intent_type: str
    ) -> Optional[str]:
        """
        Suggest optimal timing based on patterns and intent.
        """
        if not patterns:
            return None

        peak_hour = patterns.get('peak_hour')
        if peak_hour is None:
            return None

        # Map hour to time of day
        if 9 <= peak_hour <= 11:
            time_desc = "morning (9-11am)"
        elif 14 <= peak_hour <= 16:
            time_desc = "afternoon (2-4pm)"
        else:
            time_desc = f"around {peak_hour}:00"

        # Urgent messages should go immediately
        if intent_type == "urgent":
            return "immediately (urgent)"

        return f"Best engagement during {time_desc}"

    async def enrich_message(
        self,
        message: str,
        context: CommunicationContext
    ) -> str:
        """
        Enrich message with relevant context from Memory and KG.

        Example:
        - Input: "Campaign results are ready"
        - Output: "Q4 Campaign results are ready (launched Oct 5th)"
        """
        enriched = message

        # Add relevant memory context if available
        if context.recent_context:
            # Find most relevant memory
            most_relevant = max(
                context.recent_context,
                key=lambda m: m.get('importance', 0)
            )

            # Extract key date or fact
            memory_content = most_relevant.get('content', '')
            if memory_content:
                # Simple enrichment: append context hint
                # (Could be more sophisticated with LLM)
                date_str = most_relevant.get('date', '')
                if date_str:
                    try:
                        date_obj = datetime.fromisoformat(date_str)
                        date_display = date_obj.strftime("%b %d")
                        enriched += f" (ref: {date_display})"
                    except:
                        pass

        return enriched


def get_communication_context_service(supabase: Client) -> CommunicationContextService:
    """Factory function to get CommunicationContextService instance."""
    return CommunicationContextService(supabase)
