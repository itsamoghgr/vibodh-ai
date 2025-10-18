"""
Structured Logging Configuration
"""

import logging
import sys
from typing import Any
import json
from datetime import datetime
from .config import settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def setup_logging() -> logging.Logger:
    """Setup application logging"""

    # Create logger
    logger = logging.getLogger("vibodh")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Remove existing handlers
    logger.handlers = []

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)

    # Set formatter based on config
    if settings.LOG_FORMAT == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    logger.addHandler(handler)

    return logger


# Create logger instance
logger = setup_logging()


def log_request(method: str, path: str, **kwargs: Any) -> None:
    """Log HTTP request"""
    logger.info(f"{method} {path}", extra={"type": "request", **kwargs})


def log_response(method: str, path: str, status_code: int, duration_ms: float, **kwargs: Any) -> None:
    """Log HTTP response"""
    logger.info(
        f"{method} {path} - {status_code}",
        extra={
            "type": "response",
            "status_code": status_code,
            "duration_ms": duration_ms,
            **kwargs,
        },
    )


def log_error(error: Exception, context: str = "", **kwargs: Any) -> None:
    """Log error with context"""
    logger.error(
        f"{context}: {str(error)}",
        exc_info=True,
        extra={"type": "error", "error_type": type(error).__name__, **kwargs},
    )


def log_service_call(service: str, method: str, **kwargs: Any) -> None:
    """Log service method call"""
    logger.debug(
        f"[{service}] {method}",
        extra={"type": "service_call", "service": service, "method": method, **kwargs},
    )
