# Communication Agent Tests

Comprehensive test suite for the enhanced CommunicationAgent and supporting services.

## Test Structure

```
tests/
├── services/
│   ├── test_communication_reasoning_service.py    # Reasoning service tests
│   ├── test_communication_context_service.py      # Context service tests
│   └── test_communication_event_listener.py       # Event listener tests
├── agents/
│   └── test_communication_agent_integration.py    # Full lifecycle integration tests
└── README.md
```

## Test Coverage

### CommunicationReasoningService Tests
- **Intent Classification**: Urgent, Strategic, Routine, Informational
- **Module Recommendation**: RAG, KG, Memory, Insight
- **Audience Identification**: Channel extraction, team detection
- **Verification Service**: Business hours, frequency checks, LLM verification
- **LLM Integration**: Classification success/fallback, verification
- **Error Handling**: Graceful degradation

### CommunicationContextService Tests
- **Topic Extraction**: Pattern matching, keyword fallback
- **Audience Identification**: Channels, teams, people from KG
- **Memory Context**: Recent context gathering, filtering, truncation
- **Related Entities**: Entity discovery from KG
- **Communication Patterns**: Peak hour analysis, activity trends
- **Timing Suggestions**: Morning, afternoon, urgent
- **Message Enrichment**: Context-aware message enhancement

### CommunicationEventListener Tests
- **Insight Events**: High-importance, strategic, actionable, duplicate suppression
- **Action Plan Events**: Strategic completion, low-risk filtering
- **System Events**: Critical errors, duplicate suppression
- **Learning Events**: Milestone detection, performance tracking
- **Communication Triggering**: Event creation, metadata handling
- **Message Formatting**: Insights, plans, system events, milestones

### CommunicationAgent Integration Tests
- **Observe Phase**: User requests, pending events, no action
- **Plan Phase**: Informational (1 step), Strategic (3 steps), Urgent (2 steps)
- **Execute Phase**: Verification pass/block, Slack posting
- **Reflect Phase**: Success/failure analysis, engagement measurement, adaptive learning
- **Complete Lifecycle**: Full Observe → Plan → Execute → Reflect workflow

## Running Tests

### Install Dependencies

```bash
cd /Users/amoghramagiri/Documents/Projects/vibodh/vibodh-ai
pip install pytest pytest-asyncio pytest-mock
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Reasoning service tests
pytest tests/services/test_communication_reasoning_service.py

# Context service tests
pytest tests/services/test_communication_context_service.py

# Event listener tests
pytest tests/services/test_communication_event_listener.py

# Integration tests
pytest tests/agents/test_communication_agent_integration.py
```

### Run by Test Class

```bash
# Run only intent classification tests
pytest tests/services/test_communication_reasoning_service.py::TestIntentClassification

# Run only audience identification tests
pytest tests/services/test_communication_context_service.py::TestAudienceIdentification
```

### Run by Test Name

```bash
# Run specific test
pytest tests/services/test_communication_reasoning_service.py::TestIntentClassification::test_classify_urgent_intent
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage Report

```bash
pytest --cov=app --cov-report=html
# Open htmlcov/index.html to view coverage report
```

## Test Markers

Tests are marked for organization:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only async tests
pytest -m asyncio

# Skip slow tests
pytest -m "not slow"
```

## Writing New Tests

### Test Template

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

@pytest.fixture
def mock_service():
    """Create mock service."""
    return Mock()

class TestFeature:
    """Tests for specific feature."""

    def test_feature_success(self, mock_service):
        """Test successful feature execution."""
        # Arrange
        expected = "result"

        # Act
        result = mock_service.method()

        # Assert
        assert result == expected

    @pytest.mark.asyncio
    async def test_async_feature(self, mock_service):
        """Test async feature."""
        result = await mock_service.async_method()
        assert result is not None
```

### Mocking Best Practices

1. **Mock External Dependencies**: Always mock Supabase, external APIs, file I/O
2. **Use AsyncMock for Async**: Use `AsyncMock` for async methods
3. **Patch at Import Point**: Patch where the object is used, not defined
4. **Return Values**: Set `return_value` for sync, use `AsyncMock()` for async

### Test Organization

- **Arrange**: Set up test data and mocks
- **Act**: Execute the code being tested
- **Assert**: Verify expected outcomes

## Continuous Integration

Add to CI/CD pipeline:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-asyncio pytest-mock
      - run: pytest
```

## Test Statistics

- **Total Tests**: 80+
- **Services Tests**: 60+
- **Integration Tests**: 20+
- **Coverage Target**: 90%+

## Troubleshooting

### Import Errors

If you get import errors, ensure the project root is in PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/Users/amoghramagiri/Documents/Projects/vibodh/vibodh-ai"
pytest
```

### Async Warnings

If you see async warnings, ensure `pytest-asyncio` is installed:

```bash
pip install pytest-asyncio
```

### Mock Issues

If mocks aren't working, verify the patch path:

```python
# Wrong (patching at definition)
with patch('app.services.service.Method'):

# Right (patching at usage)
with patch('app.agents.agent.Method'):
```

## Performance

Expected test execution time:
- **Unit Tests**: ~10-30 seconds
- **Integration Tests**: ~30-60 seconds
- **Full Suite**: ~1-2 minutes

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure 90%+ coverage for new code
3. Add integration test for complete workflows
4. Update this README with new test information
