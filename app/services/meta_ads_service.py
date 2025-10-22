"""
Meta Ads Service - Phase 6
Mock implementation of Meta Marketing API (Facebook/Instagram Ads) for development

In mock mode, simulates Meta Marketing API responses with realistic synthetic data.
Designed to be easily swappable with real Facebook Business SDK when ready.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from supabase import Client
from app.core.config import settings
from app.core.logging import logger
from app.services.ads_synthetic_data_generator import get_synthetic_generator
import time


class MetaAdsService:
    """
    Meta Marketing API client (mock mode).

    In production, this would use facebook_business library to interact with Meta Marketing API.
    For development, generates realistic synthetic data.

    API Documentation:
    https://developers.facebook.com/docs/marketing-apis
    """

    def __init__(self, supabase: Client, mock_mode: bool = True):
        """
        Initialize Meta Ads service.

        Args:
            supabase: Supabase client for data storage
            mock_mode: If True, use synthetic data generator instead of real API
        """
        self.supabase = supabase
        self.mock_mode = mock_mode or settings.ADS_MOCK_MODE
        self.generator = get_synthetic_generator() if self.mock_mode else None

        if not self.mock_mode:
            # In production, initialize real Facebook Business SDK
            # from facebook_business.api import FacebookAdsApi
            # FacebookAdsApi.init(
            #     app_id=settings.META_ADS_APP_ID,
            #     app_secret=settings.META_ADS_APP_SECRET,
            #     access_token=access_token
            # )
            logger.warning("Meta Ads Service initialized in PRODUCTION mode but not implemented yet")

        logger.info(f"Meta Ads Service initialized (mock_mode={self.mock_mode})")

    def get_oauth_authorization_url(self, org_id: str, redirect_uri: str) -> str:
        """
        Generate OAuth 2.0 authorization URL for user to grant access.

        Args:
            org_id: Organization ID
            redirect_uri: Callback URL after authorization

        Returns:
            Authorization URL
        """
        if self.mock_mode:
            # Return mock URL
            return f"https://www.facebook.com/v18.0/dialog/oauth?mock=true&org_id={org_id}&redirect_uri={redirect_uri}"

        # In production:
        # scope = "ads_read,ads_management,business_management"
        # return f"https://www.facebook.com/v18.0/dialog/oauth?client_id={settings.META_ADS_APP_ID}&redirect_uri={redirect_uri}&scope={scope}"
        raise NotImplementedError("Production OAuth not implemented")

    def exchange_authorization_code(
        self,
        authorization_code: str,
        redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Authorization code from OAuth callback
            redirect_uri: Redirect URI used in authorization

        Returns:
            Token response with access_token, expires_in
        """
        if self.mock_mode:
            # Return mock tokens
            return {
                "access_token": f"mock_meta_ads_access_token_{int(time.time())}",
                "token_type": "bearer",
                "expires_in": 5183944  # ~60 days (Meta's default)
            }

        # In production:
        # Make request to https://graph.facebook.com/v18.0/oauth/access_token
        raise NotImplementedError("Production OAuth not implemented")

    def exchange_short_lived_for_long_lived_token(
        self,
        short_lived_token: str
    ) -> Dict[str, Any]:
        """
        Exchange short-lived token for long-lived token (60 days).

        Args:
            short_lived_token: Short-lived access token

        Returns:
            Long-lived token response
        """
        if self.mock_mode:
            return {
                "access_token": f"mock_long_lived_token_{int(time.time())}",
                "token_type": "bearer",
                "expires_in": 5183944  # ~60 days
            }

        # In production:
        # Make request to /oauth/access_token with grant_type=fb_exchange_token
        raise NotImplementedError("Production token exchange not implemented")

    def list_ad_accounts(self, access_token: str) -> List[Dict[str, Any]]:
        """
        List all ad accounts accessible by the authenticated user.

        Args:
            access_token: OAuth access token

        Returns:
            List of ad account dictionaries
        """
        if self.mock_mode:
            # Generate 1-2 mock ad accounts
            num_accounts = 2
            accounts = []

            for i in range(num_accounts):
                account = {
                    "account_id": f"act_{100000000000 + i}",
                    "id": f"act_{100000000000 + i}",
                    "name": f"Meta Ads Account {i + 1}",
                    "account_status": 1,  # 1 = ACTIVE
                    "currency": "USD",
                    "timezone_name": "America/Los_Angeles",
                    "timezone_offset_hours_utc": -8,
                    "business_name": f"Business {i + 1}",
                    "business_id": f"{900000000000 + i}",
                    "is_prepay_account": False,
                    "amount_spent": 0,
                    "balance": 0
                }
                accounts.append(account)

            logger.info(f"[MOCK] Listed {len(accounts)} Meta ad accounts")
            return accounts

        # In production:
        # from facebook_business.adobjects.user import User
        # from facebook_business.adobjects.adaccount import AdAccount
        # me = User(fbid='me')
        # accounts = me.get_ad_accounts(fields=[
        #     AdAccount.Field.account_id,
        #     AdAccount.Field.name,
        #     AdAccount.Field.currency,
        #     AdAccount.Field.account_status
        # ])
        # return [account.export_all_data() for account in accounts]
        raise NotImplementedError("Production account listing not implemented")

    def get_campaigns(
        self,
        access_token: str,
        ad_account_id: str,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Fetch all campaigns for a Meta ad account.

        Args:
            access_token: OAuth access token
            ad_account_id: Meta ad account ID (format: act_XXXXX)
            include_archived: Include archived campaigns

        Returns:
            List of campaign dictionaries
        """
        if self.mock_mode:
            logger.info(f"[MOCK] Fetching campaigns for account {ad_account_id}")

            # Generate 3-6 campaigns per account
            num_campaigns = 5
            campaigns = []

            for i in range(num_campaigns):
                campaign_data = self.generator.generate_campaigns(
                    account_id=ad_account_id,
                    org_id="mock_org",
                    platform="meta_ads",
                    num_campaigns=1
                )[0]

                # Map status to Meta's status codes
                status_map = {
                    "active": "ACTIVE",
                    "paused": "PAUSED",
                    "deleted": "ARCHIVED",
                    "ended": "ARCHIVED"
                }

                # Format to match Meta Marketing API response structure
                campaign = {
                    "id": campaign_data["campaign_id"],
                    "name": campaign_data["campaign_name"],
                    "status": status_map.get(campaign_data["status"], "ACTIVE"),
                    "effective_status": status_map.get(campaign_data["status"], "ACTIVE"),
                    "objective": campaign_data["objective"].upper(),
                    "budget_remaining": int(campaign_data["budget_amount"] * 100),
                    "daily_budget": int(campaign_data["budget_amount"] * 100) if campaign_data["budget_type"] == "daily" else None,
                    "lifetime_budget": int(campaign_data["budget_amount"] * 100) if campaign_data["budget_type"] == "lifetime" else None,
                    "start_time": f"{campaign_data['start_date']}T00:00:00+0000",
                    "stop_time": f"{campaign_data['end_date']}T23:59:59+0000" if campaign_data["end_date"] else None,
                    "buying_type": "AUCTION",
                    "bid_strategy": campaign_data["bid_strategy"].upper(),
                    "special_ad_categories": [],
                    "created_time": f"{campaign_data['start_date']}T00:00:00+0000",
                    "updated_time": datetime.now().isoformat()
                }

                campaigns.append(campaign)

            logger.info(f"[MOCK] Fetched {len(campaigns)} campaigns")
            return campaigns

        # In production:
        # from facebook_business.adobjects.adaccount import AdAccount
        # from facebook_business.adobjects.campaign import Campaign
        # account = AdAccount(ad_account_id)
        # campaigns = account.get_campaigns(fields=[
        #     Campaign.Field.id,
        #     Campaign.Field.name,
        #     Campaign.Field.status,
        #     Campaign.Field.objective,
        #     Campaign.Field.daily_budget,
        #     Campaign.Field.lifetime_budget,
        #     Campaign.Field.start_time,
        #     Campaign.Field.stop_time
        # ])
        # return [campaign.export_all_data() for campaign in campaigns]
        raise NotImplementedError("Production campaign fetching not implemented")

    def get_campaign_insights(
        self,
        access_token: str,
        campaign_id: str,
        start_date: date,
        end_date: date,
        breakdown: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Fetch performance insights (metrics) for a specific campaign.

        Args:
            access_token: OAuth access token
            campaign_id: Campaign ID
            start_date: Start date for insights
            end_date: End date for insights
            breakdown: Time breakdown ("day", "week", "month")

        Returns:
            List of daily/weekly/monthly insight dictionaries
        """
        if self.mock_mode:
            logger.info(
                f"[MOCK] Fetching insights for campaign {campaign_id} "
                f"from {start_date} to {end_date}"
            )

            # Generate synthetic campaign for metrics
            mock_campaign = {
                "platform": "meta_ads",
                "campaign_id": campaign_id,
                "start_date": (datetime.now() - timedelta(days=90)).date().isoformat(),
                "end_date": None,
                "budget_amount": 75.0,
                "budget_type": "daily",
                "metadata": {
                    "campaign_type": "feed",
                    "trend": "stable"
                }
            }

            # Generate metrics
            days = (end_date - start_date).days + 1
            metrics = self.generator.generate_metrics(mock_campaign, days=days)

            # Filter to requested date range and format as Meta insights
            insights = []
            for metric in metrics:
                metric_date = datetime.fromisoformat(metric["metric_date"]).date()
                if start_date <= metric_date <= end_date:
                    insight = {
                        "date_start": metric["metric_date"],
                        "date_stop": metric["metric_date"],
                        "impressions": str(metric["impressions"]),
                        "clicks": str(metric["clicks"]),
                        "ctr": str(metric["ctr"]),
                        "spend": str(metric["spend"]),
                        "reach": str(metric["reach"]) if metric.get("reach") else "0",
                        "frequency": str(metric["frequency"]) if metric.get("frequency") else "1",
                        "actions": [
                            {"action_type": "post_engagement", "value": str(metric.get("engagement_count", 0))},
                            {"action_type": "link_click", "value": str(int(metric["clicks"] * 0.8))},
                            {"action_type": "post_reaction", "value": str(int(metric.get("engagement_count", 0) * 0.3))},
                            {"action_type": "offsite_conversion", "value": str(metric.get("conversions", 0))}
                        ] if metric.get("engagement_count") else [],
                        "action_values": [
                            {"action_type": "offsite_conversion", "value": str(metric.get("conversion_value", 0))}
                        ] if metric.get("conversion_value") else [],
                        "cost_per_action_type": [
                            {"action_type": "offsite_conversion", "value": str(metric.get("cpa", 0))}
                        ] if metric.get("cpa") else [],
                        "cpc": str(metric["cpc"]),
                        "cpm": str(metric["cpm"]),
                        "video_30_sec_watched_actions": [
                            {"action_type": "video_view", "value": str(metric.get("video_completions", 0))}
                        ] if metric.get("video_completions") else []
                    }
                    insights.append(insight)

            logger.info(f"[MOCK] Generated {len(insights)} daily insights")
            return insights

        # In production:
        # from facebook_business.adobjects.campaign import Campaign
        # from facebook_business.adobjects.adsinsights import AdsInsights
        # campaign = Campaign(campaign_id)
        # insights = campaign.get_insights(fields=[
        #     AdsInsights.Field.impressions,
        #     AdsInsights.Field.clicks,
        #     AdsInsights.Field.ctr,
        #     AdsInsights.Field.spend,
        #     AdsInsights.Field.reach,
        #     AdsInsights.Field.frequency,
        #     AdsInsights.Field.actions,
        #     AdsInsights.Field.cpc,
        #     AdsInsights.Field.cpm
        # ], params={
        #     'time_range': {'since': start_date.isoformat(), 'until': end_date.isoformat()},
        #     'time_increment': 1,  # daily
        #     'level': 'campaign'
        # })
        # return [insight.export_all_data() for insight in insights]
        raise NotImplementedError("Production insights fetching not implemented")

    def get_account_insights_summary(
        self,
        access_token: str,
        ad_account_id: str,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """
        Get aggregated insights summary for entire ad account.

        Args:
            access_token: OAuth access token
            ad_account_id: Meta ad account ID
            start_date: Start date
            end_date: End date

        Returns:
            Aggregated insights dictionary
        """
        if self.mock_mode:
            # Generate summary by aggregating campaign insights
            campaigns = self.get_campaigns(access_token, ad_account_id)

            total_impressions = 0
            total_clicks = 0
            total_spend = 0
            total_reach = 0
            total_engagement = 0
            total_conversions = 0
            total_conversion_value = 0

            for campaign in campaigns:
                if campaign["status"] != "ARCHIVED":
                    insights = self.get_campaign_insights(
                        access_token, campaign["id"],
                        start_date, end_date
                    )

                    for insight in insights:
                        total_impressions += int(insight["impressions"])
                        total_clicks += int(insight["clicks"])
                        total_spend += float(insight["spend"])
                        total_reach += int(insight.get("reach", 0))

                        # Extract engagement from actions
                        for action in insight.get("actions", []):
                            if action["action_type"] == "post_engagement":
                                total_engagement += int(action["value"])
                            elif action["action_type"] == "offsite_conversion":
                                total_conversions += int(action["value"])

                        # Extract conversion value
                        for value in insight.get("action_values", []):
                            if value["action_type"] == "offsite_conversion":
                                total_conversion_value += float(value["value"])

            avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
            avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0
            avg_cpm = (total_spend / total_impressions * 1000) if total_impressions > 0 else 0
            avg_frequency = (total_impressions / total_reach) if total_reach > 0 else 1
            engagement_rate = (total_engagement / total_impressions * 100) if total_impressions > 0 else 0
            roas = (total_conversion_value / total_spend) if total_spend > 0 else 0

            return {
                "account_id": ad_account_id,
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "metrics": {
                    "impressions": total_impressions,
                    "clicks": total_clicks,
                    "ctr": round(avg_ctr, 2),
                    "spend": round(total_spend, 2),
                    "reach": total_reach,
                    "frequency": round(avg_frequency, 2),
                    "engagement": total_engagement,
                    "engagement_rate": round(engagement_rate, 2),
                    "conversions": total_conversions,
                    "conversion_value": round(total_conversion_value, 2),
                    "avg_cpc": round(avg_cpc, 2),
                    "avg_cpm": round(avg_cpm, 2),
                    "roas": round(roas, 2)
                },
                "campaigns_count": len([c for c in campaigns if c["status"] != "ARCHIVED"])
            }

        raise NotImplementedError("Production account summary not implemented")

    def validate_connection(self, access_token: str) -> bool:
        """
        Validate that the access token is valid and has necessary permissions.

        Args:
            access_token: OAuth access token

        Returns:
            True if valid, False otherwise
        """
        if self.mock_mode:
            # Mock validation always succeeds
            logger.info("[MOCK] Validating connection - always returns True in mock mode")
            return True

        try:
            # In production, try to get user info or list ad accounts
            # self.list_ad_accounts(access_token)
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False


# Singleton instance
_meta_ads_service: Optional[MetaAdsService] = None


def get_meta_ads_service(supabase: Client) -> MetaAdsService:
    """Get or create singleton Meta Ads service instance."""
    global _meta_ads_service
    if _meta_ads_service is None:
        _meta_ads_service = MetaAdsService(supabase)
    return _meta_ads_service
