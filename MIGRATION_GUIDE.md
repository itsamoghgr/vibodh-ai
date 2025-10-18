# Migration Guide: Vibodh AI Backend Restructuring

## Overview

This guide explains how to migrate from the old monolithic structure to the new modular architecture.

## What Changed?

### Old Structure
```
vibodh-ai/
├── main.py                 # 2000+ lines, all routes in one file
├── services/              # Service classes
├── sql/                   # Database migrations
└── requirements.txt
```

### New Structure
```
vibodh-ai/
├── app/
│   ├── main.py           # Clean, minimal entry point
│   ├── core/             # Configuration & logging
│   ├── db/               # Database layer
│   ├── models/           # Pydantic models
│   ├── services/         # Business logic
│   ├── connectors/       # Integration abstractions
│   ├── api/v1/           # Versioned API routes
│   ├── utils/            # Utilities
│   └── tests/            # Test suite
├── Dockerfile
├── docker-compose.yml
└── pytest.ini
```

## Step-by-Step Migration

### 1. Update Import Statements

**Before:**
```python
from services.rag_service import get_rag_service
from services.orchestrator_service import get_orchestrator_service
```

**After:**
```python
from app.services import get_rag_service, get_orchestrator_service
```

### 2. Update Configuration Access

**Before:**
```python
import os
supabase_url = os.getenv("SUPABASE_URL")
```

**After:**
```python
from app.core.config import settings
supabase_url = settings.SUPABASE_URL
```

### 3. Update Database Access

**Before:**
```python
from supabase import create_client
supabase = create_client(url, key)
```

**After:**
```python
from app.db import supabase  # Singleton instance
# Or create new instance:
from app.db import get_supabase_client
supabase = get_supabase_client()
```

### 4. Update Logging

**Before:**
```python
print(f"[SERVICE] Processing...")
```

**After:**
```python
from app.core.logging import logger
logger.info("Processing", extra={"service": "rag"})
```

### 5. Update API Endpoints

All endpoints are now versioned under `/api/v1/`:

**Before:**
```
POST /api/orchestrate/query
GET /api/rag/search
```

**After:**
```
POST /api/v1/orchestrate/query
GET /api/v1/rag/search
```

### 6. Update Docker Usage

**Before:**
```bash
# No Docker support
python main.py
```

**After:**
```bash
# Development
python -m app.main

# Production (Docker)
docker-compose up --build
```

## Environment Variables

No changes required to `.env` file - all existing variables work with the new `settings` object.

## Running the Application

### Development Mode

```bash
# Option 1: Run as module
python -m app.main

# Option 2: Using uvicorn
uvicorn app.main:app --reload

# Option 3: Keep old main.py for compatibility
python main.py  # If you create a wrapper
```

### Production Mode

```bash
# Docker Compose (recommended)
docker-compose up -d

# Or manual Docker
docker build -t vibodh-ai .
docker run -p 8000:8000 --env-file .env vibodh-ai
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test types
pytest app/tests/unit/
pytest app/tests/integration/
```

## Breaking Changes

### API Routes

All routes now have `/v1/` prefix:
- `POST /api/orchestrate/query` → `POST /api/v1/orchestrate/query`
- `GET /api/rag/search` → `GET /api/v1/rag/search`

### Import Paths

All imports must use `app.` prefix:
- `from services.rag_service import...` → `from app.services import...`
- `from config import settings` → `from app.core.config import settings`

### Configuration

Environment variables are now accessed via `settings` object instead of `os.getenv()`.

## Rollback Plan

If you need to rollback:

1. The old services are still in `app/services/` (just moved, not changed)
2. SQL migrations are in `app/db/migrations/` (just moved from `/sql`)
3. Keep the old `main.py` as a wrapper if needed

## Common Issues & Solutions

### Issue: Import errors

**Error:** `ModuleNotFoundError: No module named 'app'`

**Solution:** Run from project root: `python -m app.main`

### Issue: Cannot find .env

**Error:** Settings validation error

**Solution:** Ensure `.env` is in project root, or set `ENV_FILE` path

### Issue: Services not found

**Error:** `No module named 'services'`

**Solution:** Update imports to `from app.services import...`

## Verification Checklist

After migration, verify:

- [ ] Application starts without errors: `python -m app.main`
- [ ] Health check works: `curl http://localhost:8000/health`
- [ ] API docs accessible: `http://localhost:8000/docs` (if DEBUG=true)
- [ ] Database connections work
- [ ] All integrations (Slack, ClickUp) functional
- [ ] Tests pass: `pytest`
- [ ] Docker build succeeds: `docker build -t vibodh-ai .`
- [ ] Docker run succeeds: `docker-compose up`

## Benefits of New Architecture

1. **Modularity**: Clean separation of concerns
2. **Testability**: Easier to write and maintain tests
3. **Scalability**: Can add new features without touching core
4. **Type Safety**: Pydantic models ensure data validation
5. **API Versioning**: Can maintain v1 while building v2
6. **Docker Ready**: Production-grade containerization
7. **Better Logging**: Structured, JSON-formatted logs
8. **Configuration Management**: Centralized, type-safe config

## Next Steps

1. Update frontend to use `/api/v1/` endpoints
2. Add CI/CD pipeline (GitHub Actions)
3. Add monitoring (Prometheus metrics)
4. Implement caching layer (Redis)
5. Add rate limiting

## Support

For questions or issues during migration:
1. Check `ARCHITECTURE.md` for detailed structure docs
2. Review example code in each module
3. Open an issue on GitHub

---

**Migration Date:** 2025-10-18
**New Version:** 1.0.0 (Modular Architecture)
