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

    yield

    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")


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
