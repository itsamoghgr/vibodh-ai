"""
Vibodh AI - Main Application
Production-grade modular FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.logging import logger
from app.api.v1 import api_v1_router
from app.db import supabase
from app.services.agent_registry import get_agent_registry
from app.agents import MarketingAgent, CommunicationAgent
from app.workers import start_cil_worker, stop_cil_worker, start_ads_worker, stop_ads_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Register agents (Phase 4)
    try:
        agent_registry = get_agent_registry(supabase)

        # Register CommunicationAgent class (simple messaging)
        agent_registry.register_agent_class("communication_agent", CommunicationAgent)
        logger.info("Registered CommunicationAgent with framework")

        # Register MarketingAgent class (complex campaigns)
        agent_registry.register_agent_class("marketing_agent", MarketingAgent)
        logger.info("Registered MarketingAgent with framework")

        # TODO: Register other agents as they are implemented
        # agent_registry.register_agent_class("ops_agent", OpsAgent)
        # agent_registry.register_agent_class("hr_agent", HRAgent)

    except Exception as e:
        logger.error(f"Failed to register agents: {e}")

    # Start CIL worker (Phase 5)
    try:
        import os
        # Check if CIL is enabled via environment variable
        cil_enabled = os.getenv("CIL_ENABLED", "true").lower() == "true"

        if cil_enabled:
            start_cil_worker(
                learning_cycle_time=os.getenv("CIL_LEARNING_CYCLE_TIME", "2:00"),  # Default 2 AM UTC
                telemetry_interval_minutes=int(os.getenv("CIL_TELEMETRY_INTERVAL_MINUTES", "5")),
                proposal_check_interval_minutes=int(os.getenv("CIL_PROPOSAL_CHECK_INTERVAL_MINUTES", "15")),
                enabled=True
            )
            logger.info("🧠 CIL worker started successfully")
        else:
            logger.info("CIL worker disabled via configuration")
    except Exception as e:
        logger.error(f"Failed to start CIL worker: {e}")

    # Start Ads worker (Phase 6)
    try:
        import os
        # Check if Ads worker is enabled via environment variable
        ads_worker_enabled = os.getenv("ADS_WORKER_ENABLED", "true").lower() == "true"

        if ads_worker_enabled:
            start_ads_worker(
                sync_interval_hours=int(os.getenv("ADS_SYNC_INTERVAL_HOURS", "1")),  # Default hourly
                anomaly_check_time=os.getenv("ADS_ANOMALY_CHECK_TIME", "4:00"),  # Default 4 AM UTC
                enabled=True
            )
            logger.info("📊 Ads worker started successfully")
        else:
            logger.info("Ads worker disabled via configuration")
    except Exception as e:
        logger.error(f"Failed to start Ads worker: {e}")

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")

    # Stop CIL worker
    try:
        stop_cil_worker()
        logger.info("CIL worker stopped")
    except Exception as e:
        logger.error(f"Error stopping CIL worker: {e}")

    # Stop Ads worker
    try:
        stop_ads_worker()
        logger.info("Ads worker stopped")
    except Exception as e:
        logger.error(f"Error stopping Ads worker: {e}")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Intelligent knowledge management and AI-powered insights platform",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_v1_router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "environment": settings.ENVIRONMENT
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
