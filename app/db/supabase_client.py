"""
Supabase Client Factory
Centralized database connection management
"""

from supabase import create_client, Client
from functools import lru_cache
from app.core.config import settings
from app.core.logging import logger


@lru_cache()
def get_supabase_client() -> Client:
    """
    Get or create Supabase client instance.
    Uses lru_cache to ensure singleton pattern.

    Note: Uses SERVICE_ROLE_KEY since SUPABASE_KEY is optional.
    In production, you may want to use SUPABASE_KEY (anon key) for client-facing operations.

    Returns:
        Client: Supabase client instance
    """
    try:
        logger.info("Initializing Supabase client")

        client = create_client(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=settings.SUPABASE_KEY or settings.SUPABASE_SERVICE_ROLE_KEY
        )

        logger.info("Supabase client initialized successfully")
        return client

    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        raise


@lru_cache()
def get_supabase_admin_client() -> Client:
    """
    Get or create Supabase admin client with service role key.
    Use this for operations that bypass RLS.

    Returns:
        Client: Supabase admin client instance
    """
    try:
        logger.info("Initializing Supabase admin client")

        client = create_client(
            supabase_url=settings.SUPABASE_URL,
            supabase_key=settings.SUPABASE_SERVICE_ROLE_KEY
        )

        logger.info("Supabase admin client initialized successfully")
        return client

    except Exception as e:
        logger.error(f"Failed to initialize Supabase admin client: {e}")
        raise


# Export default client
supabase = get_supabase_client()
supabase_admin = get_supabase_admin_client()
