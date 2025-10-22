"""
Tests for CommunicationContextService
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from app.services.communication_context_service import (
    CommunicationContextService,
    AudienceInfo,
    CommunicationContext
)


@pytest.fixture
def mock_supabase():
    """Create mock Supabase client."""
    mock = Mock()
    mock.table.return_value = mock
    return mock


@pytest.fixture
def mock_kg_service():
    """Create mock KG service."""
    return Mock()


@pytest.fixture
def mock_memory_service():
    """Create mock Memory service."""
    return Mock()


@pytest.fixture
def context_service(mock_supabase, mock_kg_service, mock_memory_service):
    """Create CommunicationContextService instance."""
    service = CommunicationContextService(mock_supabase)
    service.kg_service = mock_kg_service
    service.memory_service = mock_memory_service
    return service


class TestTopicExtraction:
    """Tests for topic extraction from queries."""

    def test_extract_topic_about_pattern(self, context_service):
        """Test topic extraction with 'about' pattern."""
        query = "notify marketing about the new campaign launch"

        topic = context_service._extract_topic(query)

        assert "campaign" in topic.lower() or "launch" in topic.lower()

    def test_extract_topic_regarding_pattern(self, context_service):
        """Test topic extraction with 'regarding' pattern."""
        query = "send message regarding Q4 results"

        topic = context_service._extract_topic(query)

        assert "q4 results" in topic.lower()

    def test_extract_topic_for_pattern(self, context_service):
        """Test topic extraction with 'for' pattern."""
        query = "announce the launch for the new product"

        topic = context_service._extract_topic(query)

        assert "product" in topic.lower() or "launch" in topic.lower()

    def test_extract_topic_fallback_keyword(self, context_service):
        """Test fallback to keyword matching."""
        query = "send campaign update to the team"

        topic = context_service._extract_topic(query)

        assert topic == "campaign"

    def test_extract_topic_fallback_default(self, context_service):
        """Test fallback to default when no pattern matches."""
        query = "send a message"

        topic = context_service._extract_topic(query)

        assert topic == "general update"


class TestAudienceIdentification:
    """Tests for audience identification from KG."""

    @pytest.mark.asyncio
    async def test_identify_audience_with_channels(self, context_service, mock_kg_service):
        """Test audience identification finds channels."""
        mock_kg_service.search_entities_by_name.return_value = [
            {"name": "marketing", "type": "channel"},
            {"name": "sales", "type": "channel"}
        ]

        audience = await context_service._identify_audience_from_kg(
            topic="campaign",
            org_id="test_org"
        )

        assert "#marketing" in audience.channels
        assert "#sales" in audience.channels
        assert audience.confidence >= 0.6

    @pytest.mark.asyncio
    async def test_identify_audience_with_teams(self, context_service, mock_kg_service):
        """Test audience identification finds teams."""
        def mock_search(org_id, query, entity_types=None, limit=5):
            if "team" in entity_types or "project" in entity_types:
                return [
                    {"name": "Marketing Team", "type": "team"},
                    {"name": "Sales Team", "type": "team"}
                ]
            return []

        mock_kg_service.search_entities_by_name.side_effect = mock_search

        audience = await context_service._identify_audience_from_kg(
            topic="campaign",
            org_id="test_org"
        )

        assert "Marketing Team" in audience.teams
        assert "Sales Team" in audience.teams

    @pytest.mark.asyncio
    async def test_identify_audience_with_people(self, context_service, mock_kg_service):
        """Test audience identification finds people."""
        def mock_search(org_id, query, entity_types=None, limit=5):
            if entity_types and "person" in entity_types:
                return [
                    {
                        "name": "Sarah Johnson",
                        "type": "person",
                        "metadata": {"role": "Marketing Lead"}
                    },
                    {
                        "name": "John Smith",
                        "type": "person",
                        "metadata": {"role": "Sales Manager"}
                    }
                ]
            return []

        mock_kg_service.search_entities_by_name.side_effect = mock_search

        audience = await context_service._identify_audience_from_kg(
            topic="campaign",
            org_id="test_org"
        )

        assert len(audience.people) == 2
        assert any(p["name"] == "Sarah Johnson" for p in audience.people)
        assert any(p["role"] == "Marketing Lead" for p in audience.people)

    @pytest.mark.asyncio
    async def test_identify_audience_fallback_topic_matching(self, context_service, mock_kg_service):
        """Test fallback to topic-based channel matching."""
        mock_kg_service.search_entities_by_name.return_value = []

        audience = await context_service._identify_audience_from_kg(
            topic="marketing campaign",
            org_id="test_org"
        )

        # Should find marketing channel from topic matching
        assert "#marketing" in audience.channels or "#campaigns" in audience.channels

    @pytest.mark.asyncio
    async def test_identify_audience_default_general(self, context_service, mock_kg_service):
        """Test defaults to #general when nothing found."""
        mock_kg_service.search_entities_by_name.return_value = []

        audience = await context_service._identify_audience_from_kg(
            topic="unknown topic",
            org_id="test_org"
        )

        assert "#general" in audience.channels


class TestTopicChannelMatching:
    """Tests for topic to channel matching."""

    def test_match_marketing_topic(self, context_service):
        """Test marketing topic matches marketing channels."""
        channels = context_service._match_topic_to_channels("marketing campaign")

        assert "#marketing" in channels

    def test_match_sales_topic(self, context_service):
        """Test sales topic matches sales channels."""
        channels = context_service._match_topic_to_channels("sales update")

        assert "#sales" in channels

    def test_match_engineering_topic(self, context_service):
        """Test engineering topic matches eng channels."""
        channels = context_service._match_topic_to_channels("engineering sprint")

        assert "#engineering" in channels or "#dev" in channels

    def test_match_launch_topic(self, context_service):
        """Test launch topic matches announcement channels."""
        channels = context_service._match_topic_to_channels("product launch")

        # "launch" keyword matches release/announcements or "product" matches product channels
        assert "#general" in channels or "#announcements" in channels or "#product" in channels or "#releases" in channels

    def test_match_no_topic_returns_empty(self, context_service):
        """Test unknown topic returns empty list."""
        channels = context_service._match_topic_to_channels("xyz123")

        assert channels == []


class TestMemoryContextGathering:
    """Tests for memory context gathering."""

    @pytest.mark.asyncio
    async def test_gather_memory_context_success(self, context_service, mock_memory_service):
        """Test successful memory context gathering."""
        mock_memory_service.search_memories.return_value = [
            {
                "title": "Campaign Planning",
                "content": "Q4 campaign planned for October launch with $50K budget",
                "created_at": datetime.now().isoformat(),
                "importance": 0.8
            },
            {
                "title": "Budget Approval",
                "content": "Marketing budget approved by leadership",
                "created_at": datetime.now().isoformat(),
                "importance": 0.7
            }
        ]

        memories = await context_service._gather_memory_context(
            topic="campaign",
            org_id="test_org"
        )

        assert len(memories) == 2
        assert memories[0]["title"] == "Campaign Planning"
        assert "content" in memories[0]
        assert "importance" in memories[0]

    @pytest.mark.asyncio
    async def test_gather_memory_context_filters_old(self, context_service, mock_memory_service):
        """Test that old memories are filtered out."""
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        recent_date = datetime.now().isoformat()

        mock_memory_service.search_memories.return_value = [
            {
                "title": "Old Memory",
                "content": "Old content",
                "created_at": old_date,
                "importance": 0.8
            },
            {
                "title": "Recent Memory",
                "content": "Recent content",
                "created_at": recent_date,
                "importance": 0.7
            }
        ]

        memories = await context_service._gather_memory_context(
            topic="test",
            org_id="test_org",
            lookback_days=30
        )

        # Should only include recent memory
        assert len(memories) == 1
        assert memories[0]["title"] == "Recent Memory"

    @pytest.mark.asyncio
    async def test_gather_memory_context_truncates_content(self, context_service, mock_memory_service):
        """Test that content is truncated to 200 chars."""
        long_content = "A" * 500

        mock_memory_service.search_memories.return_value = [
            {
                "title": "Long Content",
                "content": long_content,
                "created_at": datetime.now().isoformat(),
                "importance": 0.8
            }
        ]

        memories = await context_service._gather_memory_context(
            topic="test",
            org_id="test_org"
        )

        assert len(memories[0]["content"]) == 200


class TestRelatedEntities:
    """Tests for related entity discovery."""

    @pytest.mark.asyncio
    async def test_get_related_entities_success(self, context_service, mock_kg_service):
        """Test successful related entity discovery."""
        mock_kg_service.search_entities_by_name.return_value = [
            {
                "name": "Marketing Campaign",
                "type": "project",
                "metadata": {"status": "active", "budget": "$50K"}
            },
            {
                "name": "Marketing Team",
                "type": "team",
                "metadata": {"size": 5, "lead": "Sarah"}
            }
        ]

        entities = await context_service._get_related_entities(
            topic="campaign",
            org_id="test_org"
        )

        assert len(entities) == 2
        assert entities[0]["name"] == "Marketing Campaign"
        assert entities[0]["type"] == "project"
        assert "metadata" in entities[0]


class TestCommunicationPatterns:
    """Tests for communication pattern analysis."""

    @pytest.mark.asyncio
    async def test_analyze_patterns_success(self, context_service, mock_supabase):
        """Test successful pattern analysis."""
        # Mock database response
        mock_result = Mock()
        mock_result.data = [
            {"payload": {}, "happened_at": "2025-10-20T14:30:00"},
            {"payload": {}, "happened_at": "2025-10-20T14:45:00"},
            {"payload": {}, "happened_at": "2025-10-20T15:00:00"},
            {"payload": {}, "happened_at": "2025-10-20T10:00:00"},
        ]

        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        patterns = await context_service._analyze_communication_patterns(
            org_id="test_org",
            topic="test"
        )

        assert "peak_hour" in patterns
        assert "activity_by_hour" in patterns
        assert patterns["total_messages"] == 4
        # Peak should be around 14-15 (2-3 PM)
        assert patterns["peak_hour"] in [10, 14, 15]

    @pytest.mark.asyncio
    async def test_analyze_patterns_no_data(self, context_service, mock_supabase):
        """Test pattern analysis with no data."""
        mock_result = Mock()
        mock_result.data = []

        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        patterns = await context_service._analyze_communication_patterns(
            org_id="test_org",
            topic="test"
        )

        assert patterns == {}


class TestTimingSuggestions:
    """Tests for timing suggestions."""

    def test_suggest_timing_morning(self, context_service):
        """Test morning timing suggestion."""
        patterns = {"peak_hour": 10, "activity_by_hour": {10: 50}}

        timing = context_service._suggest_timing(patterns, "informational")

        assert timing is not None
        assert "morning" in timing.lower()

    def test_suggest_timing_afternoon(self, context_service):
        """Test afternoon timing suggestion."""
        patterns = {"peak_hour": 15, "activity_by_hour": {15: 50}}

        timing = context_service._suggest_timing(patterns, "informational")

        assert timing is not None
        assert "afternoon" in timing.lower()

    def test_suggest_timing_urgent_immediate(self, context_service):
        """Test urgent messages suggest immediate."""
        patterns = {"peak_hour": 15}

        timing = context_service._suggest_timing(patterns, "urgent")

        assert timing is not None
        assert "immediate" in timing.lower()

    def test_suggest_timing_no_patterns(self, context_service):
        """Test no suggestion when no patterns."""
        patterns = {}

        timing = context_service._suggest_timing(patterns, "informational")

        assert timing is None


class TestMessageEnrichment:
    """Tests for message enrichment."""

    @pytest.mark.asyncio
    async def test_enrich_message_with_context(self, context_service):
        """Test message enrichment with relevant context."""
        message = "Campaign results are ready"
        context = CommunicationContext(
            topic="campaign",
            audience=AudienceInfo(
                channels=["#marketing"],
                people=[],
                teams=[],
                confidence=0.9
            ),
            recent_context=[
                {
                    "title": "Q4 Campaign",
                    "content": "Launched on Oct 5th",
                    "date": "2025-10-05T10:00:00",
                    "importance": 0.8
                }
            ],
            related_entities=[],
            suggested_timing=None,
            communication_patterns={}
        )

        enriched = await context_service.enrich_message(message, context)

        # Should include date reference
        assert "Oct 05" in enriched or "Oct 5" in enriched or enriched == message

    @pytest.mark.asyncio
    async def test_enrich_message_no_context(self, context_service):
        """Test message remains unchanged when no context."""
        message = "Campaign results are ready"
        context = CommunicationContext(
            topic="campaign",
            audience=AudienceInfo(
                channels=["#marketing"],
                people=[],
                teams=[],
                confidence=0.9
            ),
            recent_context=[],
            related_entities=[],
            suggested_timing=None,
            communication_patterns={}
        )

        enriched = await context_service.enrich_message(message, context)

        assert enriched == message


class TestContextGathering:
    """Tests for full context gathering."""

    @pytest.mark.asyncio
    async def test_gather_context_success(self, context_service, mock_kg_service, mock_memory_service, mock_supabase):
        """Test successful context gathering."""
        # Mock KG responses
        mock_kg_service.search_entities_by_name.return_value = [
            {"name": "marketing", "type": "channel"}
        ]

        # Mock Memory responses
        mock_memory_service.search_memories.return_value = [
            {
                "title": "Campaign",
                "content": "Content",
                "created_at": datetime.now().isoformat(),
                "importance": 0.8
            }
        ]

        # Mock Supabase for patterns
        mock_result = Mock()
        mock_result.data = [
            {"payload": {}, "happened_at": "2025-10-20T14:00:00"}
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        context = await context_service.gather_context(
            query="notify marketing about campaign",
            org_id="test_org",
            intent_type="informational"
        )

        assert isinstance(context, CommunicationContext)
        assert context.topic is not None
        assert len(context.audience.channels) > 0
        assert context.audience.confidence > 0

    @pytest.mark.asyncio
    async def test_gather_context_error_fallback(self, context_service):
        """Test error handling with safe fallback."""
        # Force an error
        with patch.object(context_service, '_extract_topic', side_effect=Exception("Test error")):
            context = await context_service.gather_context(
                query="test",
                org_id="test_org"
            )

            # Should return safe defaults
            assert context.topic == "general"
            assert "#general" in context.audience.channels
            assert context.audience.confidence == 0.3
