"""
Core Configuration using Pydantic BaseSettings
Centralizes all environment variables and configuration
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application Settings"""

    # Application
    APP_NAME: str = "Vibodh AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BACKEND_URL: str = "http://localhost:8000"

    # Supabase
    SUPABASE_URL: str
    SUPABASE_KEY: Optional[str] = None  # Not currently used
    SUPABASE_SERVICE_ROLE_KEY: str

    # OpenAI
    OPENAI_API_KEY: str

    # Groq
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Slack
    SLACK_CLIENT_ID: str
    SLACK_CLIENT_SECRET: str
    SLACK_REDIRECT_URI: str
    SLACK_WEBHOOK_URL: Optional[str] = None

    # ClickUp
    CLICKUP_CLIENT_ID: str
    CLICKUP_CLIENT_SECRET: str
    CLICKUP_REDIRECT_URI: str
    CLICKUP_WEBHOOK_URL: Optional[str] = None

    # Google Ads (Phase 6)
    GOOGLE_ADS_CLIENT_ID: str = ""
    GOOGLE_ADS_CLIENT_SECRET: str = ""
    GOOGLE_ADS_DEVELOPER_TOKEN: str = ""  # Required for production Google Ads API
    GOOGLE_ADS_REDIRECT_URI: str = ""

    # Meta Ads (Phase 6)
    META_ADS_APP_ID: str = ""
    META_ADS_APP_SECRET: str = ""
    META_ADS_REDIRECT_URI: str = ""

    # Ads Integration Settings (Phase 6)
    ADS_MOCK_MODE: bool = True  # Use synthetic data instead of real APIs
    ADS_WORKER_ENABLED: bool = True  # Enable automatic ads sync worker
    ADS_SYNC_INTERVAL_HOURS: int = 1  # How often to sync ad data
    ADS_ANOMALY_CHECK_TIME: str = "4:00"  # UTC time (HH:MM) for daily anomaly detection
    ADS_DEFAULT_LOOKBACK_DAYS: int = 90  # Default historical data window

    # SMTP Email
    SMTP_HOST: Optional[str] = ""
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = ""
    SMTP_PASSWORD: Optional[str] = ""
    FROM_EMAIL: Optional[str] = ""
    FROM_NAME: str = "Vibodh AI"
    SMTP_USE_TLS: bool = True
    SMTP_USE_SSL: bool = False

    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # JWT
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60

    # Vector Search
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    SIMILARITY_THRESHOLD: float = 0.7

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # LLM
    DEFAULT_LLM_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export settings instance
settings = get_settings()
