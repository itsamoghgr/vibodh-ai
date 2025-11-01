"""
Rate Limiter Service - Phase 6
Implements token bucket algorithm for API rate limiting

Protects against exceeding rate limits for:
- Google Ads API: 15,000 operations/day per developer token
- Meta Ads API: 200 requests/hour per ad account
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from threading import Lock
import time

from app.core.logging import logger


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    Tokens are added at a constant rate. Each API call consumes a token.
    If no tokens available, request must wait or be rejected.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = Lock()

    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: int = 1, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume
            block: If True, wait until tokens available
            timeout: Maximum time to wait (seconds)

        Returns:
            True if tokens consumed, False if rate limited
        """
        start_time = time.time()

        with self.lock:
            self._refill()

            # If enough tokens, consume and return
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            # If not blocking, reject immediately
            if not block:
                logger.warning(f"Rate limit reached: {self.tokens:.2f}/{self.capacity} tokens available")
                return False

        # Block until tokens available or timeout
        while True:
            time.sleep(0.1)  # Check every 100ms

            with self.lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning(f"Rate limit timeout: waited {timeout}s")
                return False

    def get_available_tokens(self) -> float:
        """Get current number of available tokens"""
        with self.lock:
            self._refill()
            return self.tokens

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time for tokens to be available (seconds)"""
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                return 0.0

            tokens_needed = tokens - self.tokens
            return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Global rate limiter for ads platform APIs.

    Maintains separate buckets for each platform and account.
    """

    # Rate limit configurations
    GOOGLE_ADS_DAILY_LIMIT = 15000  # operations per day
    GOOGLE_ADS_REFILL_RATE = GOOGLE_ADS_DAILY_LIMIT / 86400  # per second

    META_ADS_HOURLY_LIMIT = 200  # requests per hour per account
    META_ADS_REFILL_RATE = META_ADS_HOURLY_LIMIT / 3600  # per second

    def __init__(self):
        """Initialize rate limiter with buckets for each platform"""
        self.buckets: Dict[str, TokenBucket] = {}
        self.lock = Lock()

        logger.info("Rate limiter initialized")

    def _get_bucket_key(self, platform: str, account_id: Optional[str] = None) -> str:
        """Generate unique key for bucket"""
        if account_id:
            return f"{platform}:{account_id}"
        return platform

    def _get_or_create_bucket(self, platform: str, account_id: Optional[str] = None) -> TokenBucket:
        """Get existing bucket or create new one"""
        bucket_key = self._get_bucket_key(platform, account_id)

        with self.lock:
            if bucket_key not in self.buckets:
                # Create bucket based on platform
                if platform == "google_ads":
                    # Google Ads: single bucket for developer token
                    capacity = self.GOOGLE_ADS_DAILY_LIMIT
                    refill_rate = self.GOOGLE_ADS_REFILL_RATE
                elif platform == "meta_ads":
                    # Meta Ads: bucket per ad account
                    capacity = self.META_ADS_HOURLY_LIMIT
                    refill_rate = self.META_ADS_REFILL_RATE
                else:
                    # Default: 100 requests/minute
                    capacity = 100
                    refill_rate = 100 / 60

                self.buckets[bucket_key] = TokenBucket(capacity, refill_rate)
                logger.info(f"Created rate limit bucket: {bucket_key} (capacity={capacity}, rate={refill_rate:.4f}/s)")

            return self.buckets[bucket_key]

    def check_rate_limit(
        self,
        platform: str,
        account_id: Optional[str] = None,
        tokens: int = 1,
        block: bool = False,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Check if request is within rate limit.

        Args:
            platform: Platform name ("google_ads" or "meta_ads")
            account_id: Optional account ID for per-account limits
            tokens: Number of tokens to consume (default: 1)
            block: If True, wait for tokens to become available
            timeout: Maximum wait time in seconds

        Returns:
            True if request allowed, False if rate limited
        """
        bucket = self._get_or_create_bucket(platform, account_id)
        return bucket.consume(tokens, block, timeout)

    def get_rate_limit_status(self, platform: str, account_id: Optional[str] = None) -> Dict[str, any]:
        """
        Get current rate limit status for platform/account.

        Args:
            platform: Platform name
            account_id: Optional account ID

        Returns:
            Status dict with available tokens and wait time
        """
        bucket = self._get_or_create_bucket(platform, account_id)

        available = bucket.get_available_tokens()
        capacity = bucket.capacity
        wait_time = bucket.get_wait_time(1)

        return {
            "platform": platform,
            "account_id": account_id,
            "available_tokens": round(available, 2),
            "capacity": capacity,
            "utilization_pct": round((1 - available / capacity) * 100, 2),
            "next_token_wait_seconds": round(wait_time, 2)
        }

    def reset_bucket(self, platform: str, account_id: Optional[str] = None):
        """Reset bucket to full capacity (for testing/admin)"""
        bucket_key = self._get_bucket_key(platform, account_id)

        with self.lock:
            if bucket_key in self.buckets:
                self.buckets[bucket_key].tokens = self.buckets[bucket_key].capacity
                logger.info(f"Reset rate limit bucket: {bucket_key}")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance"""
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter()

    return _rate_limiter


# Decorator for rate-limited functions
def rate_limited(platform: str, tokens: int = 1):
    """
    Decorator to apply rate limiting to a function.

    Args:
        platform: Platform name ("google_ads" or "meta_ads")
        tokens: Number of tokens to consume per call

    Usage:
        @rate_limited("google_ads", tokens=1)
        def fetch_campaigns(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            # Extract account_id from kwargs if present
            account_id = kwargs.get('account_id') or kwargs.get('customer_id') or kwargs.get('ad_account_id')

            # Check rate limit
            if not limiter.check_rate_limit(platform, account_id, tokens, block=True, timeout=30):
                raise Exception(f"Rate limit exceeded for {platform}")

            # Execute function
            return func(*args, **kwargs)

        return wrapper
    return decorator
