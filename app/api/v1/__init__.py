"""
API v1 Router
Aggregates all API v1 route modules
"""

from fastapi import APIRouter
from .routes_orchestrator import router as orchestrator_router
from .routes_rag import router as rag_router
from .routes_kg import router as kg_router
from .routes_insights import router as insights_router
from .routes_slack import router as slack_router
from .routes_clickup import router as clickup_router
from .routes_chat import router as chat_router
from .routes_connections import router as connections_router
from .routes_documents import router as documents_router
from .routes_memory import router as memory_router
from .routes_adaptive import router as adaptive_router
from .routes_meta_learning import router as meta_learning_router

# Create main v1 router
api_v1_router = APIRouter(prefix="/api/v1")

# Include all sub-routers
api_v1_router.include_router(orchestrator_router)
api_v1_router.include_router(rag_router)
api_v1_router.include_router(kg_router)
api_v1_router.include_router(insights_router)
api_v1_router.include_router(slack_router)
api_v1_router.include_router(clickup_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(connections_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(memory_router)
api_v1_router.include_router(adaptive_router)
api_v1_router.include_router(meta_learning_router)

__all__ = ["api_v1_router"]
