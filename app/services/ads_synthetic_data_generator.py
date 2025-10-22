"""
Ads Synthetic Data Generator - Phase 6
Generates realistic advertising campaign data for testing and development

Creates synthetic but realistic metrics based on industry benchmarks:
- CTR: 0.8-2.5% (industry average)
- Quality Score: 3-10 (Google Ads scale)
- ROAS: 1.5-6.0 (healthy range)
- Engagement rates: 0.5-3.5% (social media)
"""

import random
import uuid
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
from decimal import Decimal
import math


class AdsSyntheticDataGenerator:
    """
    Generates realistic synthetic advertising data for development and testing.

    Supports both Google Ads and Meta Ads with platform-specific metrics.
    """

    # Campaign objective distributions
    CAMPAIGN_OBJECTIVES = {
        "google_ads": [
            "conversions", "traffic", "awareness", "leads", "sales",
            "app_installs", "local_actions"
        ],
        "meta_ads": [
            "conversions", "traffic", "engagement", "awareness",
            "app_installs", "lead_generation", "messages", "video_views"
        ]
    }

    # Realistic industry benchmarks
    BENCHMARKS = {
        "google_ads": {
            "search": {
                "ctr_range": (1.5, 3.5),  # Higher for search
                "quality_score_range": (5, 9),
                "cpc_range": (0.5, 3.0),
                "conversion_rate_range": (2.5, 8.0),
                "roas_range": (2.0, 6.0)
            },
            "display": {
                "ctr_range": (0.3, 1.2),  # Lower for display
                "quality_score_range": (4, 7),
                "cpc_range": (0.2, 1.5),
                "conversion_rate_range": (0.5, 2.5),
                "roas_range": (1.5, 4.0)
            },
            "video": {
                "ctr_range": (0.4, 1.8),
                "quality_score_range": (4, 8),
                "cpv_range": (0.05, 0.30),
                "video_view_rate_range": (20, 45),
                "engagement_rate_range": (1.0, 4.0)
            }
        },
        "meta_ads": {
            "feed": {
                "ctr_range": (0.8, 2.5),
                "cpc_range": (0.3, 2.0),
                "engagement_rate_range": (1.5, 4.5),
                "conversion_rate_range": (1.0, 5.0),
                "roas_range": (2.5, 7.0)
            },
            "stories": {
                "ctr_range": (1.2, 3.5),  # Higher engagement
                "cpc_range": (0.4, 2.5),
                "engagement_rate_range": (2.0, 6.0),
                "conversion_rate_range": (0.8, 4.0),
                "roas_range": (1.8, 5.5)
            },
            "reels": {
                "ctr_range": (1.5, 4.0),  # Highest engagement
                "cpv_range": (0.03, 0.20),
                "video_view_rate_range": (30, 60),
                "engagement_rate_range": (3.0, 8.0),
                "conversion_rate_range": (0.5, 3.0)
            }
        }
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator with optional random seed for reproducibility.

        Args:
            seed: Random seed for deterministic generation
        """
        if seed:
            random.seed(seed)

        self.account_counter = 0
        self.campaign_counter = 0

    def _random_decimal(self, min_val: float, max_val: float, decimals: int = 2) -> Decimal:
        """Generate random decimal in range with specified precision."""
        value = random.uniform(min_val, max_val)
        return Decimal(str(round(value, decimals)))

    def _apply_time_variance(self, base_value: float, days_ago: int, trend: str = "stable") -> float:
        """
        Apply time-based variance to simulate realistic campaign evolution.

        Args:
            base_value: Base metric value
            days_ago: How many days ago (0 = today)
            trend: "growing", "declining", "stable", "seasonal"
        """
        # Add random daily variance (±10%)
        daily_noise = random.uniform(0.9, 1.1)
        value = base_value * daily_noise

        # Apply trend
        if trend == "growing":
            # Improve over time (5% per 10 days)
            growth_factor = 1 + (0.05 * (90 - days_ago) / 10)
            value *= growth_factor
        elif trend == "declining":
            # Decline over time
            decline_factor = 1 - (0.03 * (90 - days_ago) / 10)
            value *= decline_factor
        elif trend == "seasonal":
            # Sine wave pattern (30-day cycle)
            seasonal_factor = 1 + 0.2 * math.sin((90 - days_ago) * math.pi / 15)
            value *= seasonal_factor

        return max(0, value)  # Ensure non-negative

    def generate_ad_account(
        self,
        org_id: str,
        platform: str,
        account_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a synthetic ad account.

        Args:
            org_id: Organization ID
            platform: "google_ads" or "meta_ads"
            account_name: Optional custom account name

        Returns:
            Ad account data dictionary
        """
        self.account_counter += 1

        if not account_name:
            account_name = f"{platform.replace('_', ' ').title()} Account {self.account_counter}"

        # Generate realistic account IDs
        if platform == "google_ads":
            account_id = f"{random.randint(1000000000, 9999999999)}"
        else:  # meta_ads
            account_id = f"act_{random.randint(100000000000, 999999999999)}"

        return {
            "org_id": org_id,
            "platform": platform,
            "account_id": account_id,
            "account_name": account_name,
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "timezone": random.choice(["America/New_York", "America/Los_Angeles", "UTC", "Europe/London"]),
            "status": "active",
            "metadata": {
                "generated": True,
                "created_by": "synthetic_generator",
                "account_manager": random.choice(["John Doe", "Jane Smith", "Alex Johnson"])
            }
        }

    def generate_campaigns(
        self,
        account_id: str,
        org_id: str,
        platform: str,
        num_campaigns: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic campaigns for an account.

        Args:
            account_id: Parent account UUID
            org_id: Organization ID
            platform: "google_ads" or "meta_ads"
            num_campaigns: Number of campaigns to generate

        Returns:
            List of campaign data dictionaries
        """
        campaigns = []
        objectives = self.CAMPAIGN_OBJECTIVES[platform]

        for i in range(num_campaigns):
            self.campaign_counter += 1

            objective = random.choice(objectives)
            campaign_type = self._get_campaign_type(platform, objective)

            # Generate realistic date range (campaigns can be 30-90 days old)
            days_old = random.randint(30, 90)
            start_date = datetime.now().date() - timedelta(days=days_old)

            # Some campaigns ended, some are ongoing
            if random.random() < 0.2:  # 20% ended
                end_date = start_date + timedelta(days=random.randint(14, days_old))
                status = "ended"
            else:
                end_date = None
                status = random.choice(["active"] * 7 + ["paused"] * 2 + ["deleted"])

            # Budget configuration
            budget_type = random.choice(["daily", "lifetime"])
            if budget_type == "daily":
                budget_amount = self._random_decimal(10, 500, 2)
            else:
                campaign_duration = (end_date - start_date).days if end_date else 90
                budget_amount = self._random_decimal(500, 10000, 2)

            # Generate campaign ID
            if platform == "google_ads":
                campaign_id = f"{random.randint(10000000, 99999999)}"
            else:
                campaign_id = f"{random.randint(1000000000000, 9999999999999)}"

            campaign = {
                "org_id": org_id,
                "account_id": account_id,
                "platform": platform,
                "campaign_id": campaign_id,
                "campaign_name": self._generate_campaign_name(platform, objective, i + 1),
                "objective": objective,
                "status": status,
                "budget_amount": float(budget_amount),
                "budget_type": budget_type,
                "bid_strategy": random.choice([
                    "manual_cpc", "target_cpa", "maximize_conversions",
                    "target_roas", "maximize_clicks"
                ]),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat() if end_date else None,
                "targeting": self._generate_targeting(platform),
                "google_ads_data": self._generate_google_ads_data(campaign_type) if platform == "google_ads" else None,
                "meta_ads_data": self._generate_meta_ads_data(campaign_type) if platform == "meta_ads" else None,
                "metadata": {
                    "generated": True,
                    "campaign_type": campaign_type,
                    "trend": random.choice(["stable", "growing", "declining", "seasonal"])
                }
            }

            campaigns.append(campaign)

        return campaigns

    def generate_metrics(
        self,
        campaign: Dict[str, Any],
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Generate daily metrics for a campaign over specified days.

        Args:
            campaign: Campaign data dictionary
            days: Number of days of metrics to generate

        Returns:
            List of daily metric data dictionaries
        """
        metrics = []
        platform = campaign["platform"]
        campaign_type = campaign["metadata"]["campaign_type"]
        trend = campaign["metadata"].get("trend", "stable")

        # Get benchmark ranges for this campaign type
        benchmarks = self.BENCHMARKS[platform][campaign_type]

        # Base daily budget (used to calculate impressions)
        daily_budget = float(campaign["budget_amount"])
        if campaign["budget_type"] == "lifetime":
            start_date = datetime.fromisoformat(campaign["start_date"]).date()
            end_date = datetime.fromisoformat(campaign["end_date"]).date() if campaign["end_date"] else datetime.now().date()
            campaign_days = max((end_date - start_date).days, 1)
            daily_budget = daily_budget / campaign_days

        # Generate metrics for each day
        start_date = datetime.fromisoformat(campaign["start_date"]).date()

        for day_offset in range(min(days, 90)):  # Max 90 days
            metric_date = start_date + timedelta(days=day_offset)

            # Stop if past campaign end date or future
            if campaign["end_date"] and metric_date > datetime.fromisoformat(campaign["end_date"]).date():
                break
            if metric_date > datetime.now().date():
                break

            # Calculate days ago for trend application
            days_ago = (datetime.now().date() - metric_date).days

            # Generate base metrics with time variance
            base_ctr = random.uniform(*benchmarks["ctr_range"])
            ctr = self._apply_time_variance(base_ctr, days_ago, trend)

            # Calculate impressions based on budget and CPC
            base_cpc = random.uniform(*benchmarks.get("cpc_range", (0.5, 2.0)))
            cpc = self._apply_time_variance(base_cpc, days_ago, trend)

            # Impressions = (Budget / CPC) / CTR
            clicks = int(daily_budget / max(cpc, 0.1))
            impressions = int(clicks / max(ctr / 100, 0.001))

            # Add realistic variance
            impressions = int(impressions * random.uniform(0.8, 1.2))
            clicks = int(impressions * (ctr / 100))

            # Ensure minimum activity
            impressions = max(impressions, 100)
            clicks = max(clicks, 1)

            # Recalculate CTR based on actual numbers
            actual_ctr = (clicks / impressions * 100) if impressions > 0 else 0

            # Calculate spend
            spend = clicks * cpc
            spend = min(spend, daily_budget * 1.1)  # Don't exceed budget by much

            # Conversion metrics
            base_conv_rate = random.uniform(*benchmarks.get("conversion_rate_range", (1.0, 5.0)))
            conversion_rate = self._apply_time_variance(base_conv_rate, days_ago, trend)
            conversions = int(clicks * (conversion_rate / 100))

            # Calculate CPA and ROAS
            cpa = (spend / conversions) if conversions > 0 else 0

            base_roas = random.uniform(*benchmarks.get("roas_range", (2.0, 5.0)))
            roas = self._apply_time_variance(base_roas, days_ago, trend)
            conversion_value = spend * roas if conversions > 0 else 0

            # Quality score (Google Ads)
            quality_score = None
            if platform == "google_ads":
                base_qs = random.uniform(*benchmarks.get("quality_score_range", (5, 8)))
                quality_score = self._apply_time_variance(base_qs, days_ago, trend)
                quality_score = max(1, min(10, quality_score))  # Clamp to 1-10

            # Engagement metrics (Meta Ads + Video)
            engagement_rate = None
            engagement_count = 0
            if platform == "meta_ads" or campaign_type == "video":
                base_eng = random.uniform(*benchmarks.get("engagement_rate_range", (1.5, 4.0)))
                engagement_rate = self._apply_time_variance(base_eng, days_ago, trend)
                engagement_count = int(impressions * (engagement_rate / 100))

            # Video metrics
            video_views = 0
            video_view_rate = None
            video_completions = 0
            video_completion_rate = None
            cpv = None

            if campaign_type in ["video", "reels"]:
                base_view_rate = random.uniform(*benchmarks.get("video_view_rate_range", (25, 50)))
                video_view_rate = self._apply_time_variance(base_view_rate, days_ago, trend)
                video_views = int(impressions * (video_view_rate / 100))

                video_completion_rate = random.uniform(15, 40)
                video_completions = int(video_views * (video_completion_rate / 100))

                cpv = (spend / video_views) if video_views > 0 else 0

            # Reach and frequency (Meta)
            reach = None
            frequency = None
            if platform == "meta_ads":
                # Reach is typically 60-80% of impressions
                reach = int(impressions * random.uniform(0.6, 0.8))
                frequency = impressions / reach if reach > 0 else 1

            # CPM
            cpm = (spend / impressions * 1000) if impressions > 0 else 0

            metric = {
                "metric_date": metric_date.isoformat(),
                "impressions": impressions,
                "clicks": clicks,
                "ctr": round(actual_ctr, 2),
                "spend": round(spend, 2),
                "conversions": conversions,
                "conversion_rate": round(conversion_rate, 2) if conversions > 0 else 0,
                "cpa": round(cpa, 2) if cpa > 0 else None,
                "roas": round(roas, 2) if conversions > 0 else None,
                "conversion_value": round(conversion_value, 2) if conversion_value > 0 else None,
                "quality_score": round(quality_score, 1) if quality_score else None,
                "engagement_rate": round(engagement_rate, 2) if engagement_rate else None,
                "engagement_count": engagement_count if engagement_count > 0 else None,
                "video_views": video_views if video_views > 0 else None,
                "video_view_rate": round(video_view_rate, 2) if video_view_rate else None,
                "video_completions": video_completions if video_completions > 0 else None,
                "video_completion_rate": round(video_completion_rate, 2) if video_completion_rate else None,
                "reach": reach,
                "frequency": round(frequency, 2) if frequency else None,
                "cpc": round(cpc, 2),
                "cpm": round(cpm, 2),
                "cpv": round(cpv, 2) if cpv else None,
                "google_ads_metrics": self._generate_google_specific_metrics() if platform == "google_ads" else None,
                "meta_ads_metrics": self._generate_meta_specific_metrics() if platform == "meta_ads" else None,
                "metadata": {
                    "generated": True,
                    "days_ago": days_ago
                }
            }

            metrics.append(metric)

        return metrics

    def _get_campaign_type(self, platform: str, objective: str) -> str:
        """Determine campaign subtype based on platform and objective."""
        if platform == "google_ads":
            if objective in ["conversions", "sales", "leads"]:
                return "search"
            elif objective in ["awareness"]:
                return "display"
            elif objective in ["video_views"]:
                return "video"
            else:
                return random.choice(["search", "display"])
        else:  # meta_ads
            if objective in ["video_views"]:
                return "reels"
            elif objective in ["awareness", "traffic"]:
                return random.choice(["feed", "stories"])
            else:
                return "feed"

    def _generate_campaign_name(self, platform: str, objective: str, number: int) -> str:
        """Generate realistic campaign names."""
        prefixes = {
            "conversions": ["Conv", "Sales", "Purchase"],
            "traffic": ["Traffic", "Clicks", "Visits"],
            "awareness": ["Brand", "Awareness", "Reach"],
            "engagement": ["Engagement", "Interaction", "Social"],
            "leads": ["Lead Gen", "Leads", "Sign Up"],
            "video_views": ["Video", "Views", "Watch"]
        }

        prefix = prefixes.get(objective, ["Campaign"])[0]
        platform_short = "GA" if platform == "google_ads" else "Meta"
        season = random.choice(["Q1", "Q2", "Q3", "Q4", "Spring", "Summer", "Fall", "Winter"])

        return f"{prefix} - {platform_short} {season} #{number}"

    def _generate_targeting(self, platform: str) -> Dict[str, Any]:
        """Generate realistic targeting configuration."""
        return {
            "locations": random.sample([
                "United States", "United Kingdom", "Canada", "Australia",
                "Germany", "France", "Japan", "Singapore"
            ], k=random.randint(1, 3)),
            "age_range": {
                "min": random.choice([18, 25, 35]),
                "max": random.choice([45, 55, 65])
            },
            "genders": random.choice([["all"], ["male"], ["female"], ["male", "female"]]),
            "interests": random.sample([
                "technology", "business", "finance", "health", "fitness",
                "travel", "education", "entertainment", "food"
            ], k=random.randint(2, 5)) if platform == "meta_ads" else None,
            "keywords": random.sample([
                "software", "saas", "cloud", "analytics", "marketing",
                "automation", "platform", "solution"
            ], k=random.randint(3, 7)) if platform == "google_ads" else None
        }

    def _generate_google_ads_data(self, campaign_type: str) -> Dict[str, Any]:
        """Generate Google Ads specific configuration."""
        return {
            "network_settings": {
                "search_network": campaign_type == "search",
                "display_network": campaign_type in ["display", "video"],
                "search_partners": random.choice([True, False])
            },
            "campaign_subtype": campaign_type,
            "ad_rotation": random.choice(["optimize", "rotate_indefinitely"]),
            "targeting_expansion": random.choice([True, False])
        }

    def _generate_meta_ads_data(self, campaign_type: str) -> Dict[str, Any]:
        """Generate Meta Ads specific configuration."""
        placements = {
            "feed": ["facebook_feed", "instagram_feed"],
            "stories": ["facebook_stories", "instagram_stories"],
            "reels": ["instagram_reels", "facebook_reels"]
        }

        return {
            "placements": placements.get(campaign_type, ["facebook_feed"]),
            "optimization_goal": random.choice([
                "IMPRESSIONS", "REACH", "LINK_CLICKS", "CONVERSIONS", "LANDING_PAGE_VIEWS"
            ]),
            "billing_event": random.choice(["IMPRESSIONS", "LINK_CLICKS"]),
            "pacing_type": random.choice(["standard", "accelerated"])
        }

    def _generate_google_specific_metrics(self) -> Dict[str, Any]:
        """Generate Google Ads specific metrics."""
        return {
            "search_impression_share": round(random.uniform(50, 95), 1),
            "search_top_impression_share": round(random.uniform(30, 80), 1),
            "avg_position": round(random.uniform(1.2, 4.5), 1),
            "interaction_rate": round(random.uniform(5, 15), 2)
        }

    def _generate_meta_specific_metrics(self) -> Dict[str, Any]:
        """Generate Meta Ads specific metrics."""
        return {
            "post_engagement": random.randint(50, 500),
            "link_clicks": random.randint(20, 200),
            "post_reactions": random.randint(10, 150),
            "post_shares": random.randint(5, 50),
            "post_saves": random.randint(3, 30)
        }


# Singleton instance
_generator: Optional[AdsSyntheticDataGenerator] = None


def get_synthetic_generator(seed: Optional[int] = None) -> AdsSyntheticDataGenerator:
    """Get or create singleton synthetic data generator."""
    global _generator
    if _generator is None:
        _generator = AdsSyntheticDataGenerator(seed=seed)
    return _generator
