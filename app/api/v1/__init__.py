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
from .routes_agents import router as agents_router
from .routes_marketing import router as marketing_router
from .routes_approvals import router as approvals_router
from .routes_analytics import router as analytics_router
from .routes_cil import router as cil_router
from .routes_ads import router as ads_router  # Phase 6
from .routes_ads_alerts import router as ads_alerts_router  # Phase 6.5

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
api_v1_router.include_router(agents_router)
api_v1_router.include_router(marketing_router)
api_v1_router.include_router(approvals_router)
api_v1_router.include_router(analytics_router)
api_v1_router.include_router(cil_router)
api_v1_router.include_router(ads_router)  # Phase 6
api_v1_router.include_router(ads_alerts_router)  # Phase 6.5

__all__ = ["api_v1_router"]
