# Vibodh AI - Modular Architecture

## Overview

This document describes the new modular, production-grade architecture for Vibodh AI backend.

## Project Structure

```
vibodh-ai/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   │
│   ├── core/                     # Core configuration & utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Pydantic Settings (centralized config)
│   │   └── logging.py           # Structured logging setup
│   │
│   ├── db/                       # Database layer
│   │   ├── __init__.py
│   │   ├── supabase_client.py   # Supabase client factory
│   │   └── migrations/          # SQL migration files (moved from /sql)
│   │       ├── 01_init_schema.sql
│   │       ├── 02_add_integrations.sql
│   │       └── ...
│   │
│   ├── models/                   # Pydantic data models
│   │   ├── __init__.py
│   │   ├── document.py          # Document models
│   │   ├── query.py             # Query request/response models
│   │   └── integration.py       # Integration models
│   │
│   ├── services/                 # Business logic layer
│   │   ├── __init__.py
│   │   ├── rag_service.py       # RAG (Retrieval-Augmented Generation)
│   │   ├── kg_service.py        # Knowledge Graph
│   │   ├── insight_service.py   # Analytics & Insights
│   │   ├── orchestrator_service.py  # Cognitive Core
│   │   ├── ingestion_service.py # Document ingestion
│   │   ├── embedding_service.py # OpenAI embeddings
│   │   ├── slack_service.py     # Slack API client
│   │   └── clickup_service.py   # ClickUp API client
│   │
│   ├── connectors/               # Integration abstraction layer
│   │   ├── __init__.py
│   │   ├── slack_connector.py   # Slack integration wrapper
│   │   └── clickup_connector.py # ClickUp integration wrapper
│   │
│   ├── api/                      # API routes (versioned)
│   │   ├── __init__.py
│   │   └── v1/                  # API version 1
│   │       ├── __init__.py
│   │       ├── routes_orchestrator.py  # Orchestrator endpoints
│   │       ├── routes_rag.py           # RAG endpoints
│   │       ├── routes_kg.py            # Knowledge Graph endpoints
│   │       ├── routes_insights.py      # Insights endpoints
│   │       ├── routes_slack.py         # Slack integration endpoints
│   │       └── routes_clickup.py       # ClickUp integration endpoints
│   │
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── text_processing.py   # Text chunking, cleaning, etc.
│   │
│   ├── tests/                    # Test suite
│   │   ├── __init__.py
│   │   ├── conftest.py          # Pytest fixtures
│   │   ├── unit/                # Unit tests
│   │   │   ├── __init__.py
│   │   │   └── test_text_processing.py
│   │   └── integration/         # Integration tests
│   │       └── __init__.py
│   │
│   └── workers/                  # Background workers (future)
│       └── __init__.py
│
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Docker Compose config
├── .dockerignore                 # Docker ignore patterns
├── README.md                     # Project documentation
└── ARCHITECTURE.md              # This file

```

## Key Design Principles

### 1. **Separation of Concerns**

Each module has a single, well-defined responsibility:

- **Core**: Configuration, logging, and shared utilities
- **DB**: Database connections and migrations
- **Models**: Data validation and serialization
- **Services**: Business logic and external service integrations
- **Connectors**: High-level integration abstractions
- **API**: HTTP request handling and routing
- **Utils**: Reusable utility functions

### 2. **Dependency Injection**

Services use factory functions with singleton patterns:

```python
from app.db import supabase
from app.services import get_rag_service

rag_service = get_rag_service(supabase)
```

### 3. **Configuration Management**

Centralized config using Pydantic BaseSettings:

```python
from app.core.config import settings

print(settings.SUPABASE_URL)
print(settings.OPENAI_API_KEY)
```

### 4. **API Versioning**

All endpoints are versioned under `/api/v1/`:

- `POST /api/v1/orchestrate/query`
- `GET /api/v1/rag/search`
- `GET /api/v1/kg/graph`
- `POST /api/v1/slack/connect`

### 5. **Structured Logging**

JSON-formatted logs for production:

```python
from app.core.logging import logger

logger.info("Processing query", extra={"org_id": org_id})
```

## Running the Application

### Development Mode

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
python -m app.main

# Or with uvicorn directly
uvicorn app.main:app --reload
```

### Production Mode (Docker)

```bash
# Build and run
docker-compose up --build

# Or manually
docker build -t vibodh-ai .
docker run -p 8000:8000 --env-file .env vibodh-ai
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest app/tests/unit/

# Integration tests only
pytest app/tests/integration/

# With coverage
pytest --cov=app
```

## Migration from Old Structure

The old structure had all services in the root `/services` directory and SQL files in `/sql`. The new structure:

1. **Moved services** → `app/services/`
2. **Moved SQL migrations** → `app/db/migrations/`
3. **Extracted configuration** → `app/core/config.py`
4. **Created API modules** → `app/api/v1/routes_*.py`
5. **Added connectors** → `app/connectors/`
6. **Added models** → `app/models/`

### Backward Compatibility

The old `main.py` at the root can be updated to import from the new `app.main`:

```python
# Old main.py (kept for compatibility)
from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

## API Documentation

When running in debug mode, API docs are available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

In production, these are disabled for security.

## Environment Variables

All configuration is managed through `.env`:

```env
# Application
ENVIRONMENT=production
DEBUG=false

# Database
SUPABASE_URL=...
SUPABASE_KEY=...

# AI Services
OPENAI_API_KEY=...
GROQ_API_KEY=...

# Integrations
SLACK_CLIENT_ID=...
CLICKUP_CLIENT_ID=...

# etc.
```

## Future Enhancements

1. **Workers Module**: Add Celery/RQ for background tasks
2. **Caching Layer**: Redis integration for performance
3. **API Gateway**: Kong or similar for rate limiting
4. **Monitoring**: Prometheus + Grafana metrics
5. **CI/CD**: GitHub Actions for automated testing and deployment

## Contributing

When adding new features:

1. Create service logic in `app/services/`
2. Define Pydantic models in `app/models/`
3. Add API routes in `app/api/v1/routes_*.py`
4. Write tests in `app/tests/`
5. Update this documentation

## Contact

For questions about this architecture, please refer to the team documentation or open an issue.
