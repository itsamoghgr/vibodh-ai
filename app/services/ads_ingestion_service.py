"""
Ads Ingestion Service - Phase 6
Orchestrates data ingestion from Google Ads and Meta Ads platforms

Handles:
- Account discovery and storage
- Campaign synchronization
- Daily metrics ingestion
- Job tracking and error handling
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from supabase import Client
from app.core.logging import logger
from app.services.google_ads_service import get_google_ads_service, reset_google_ads_service
from app.services.meta_ads_service import get_meta_ads_service
from app.services.ads_document_service import get_ads_document_service
from app.services.embedding_service import get_embedding_service
from app.services.memory_service import get_memory_service
import uuid
import time
import asyncio


class AdsIngestionService:
    """
    Unified ads ingestion orchestrator for Google Ads and Meta Ads.

    Coordinates data flow from external ad platforms into Supabase database.
    """

    def __init__(self, supabase: Client):
        """
        Initialize ads ingestion service.

        Args:
            supabase: Supabase client for data storage
        """
        self.supabase = supabase

        # Reset Google Ads singleton to ensure fresh instance with generator
        reset_google_ads_service()

        self.google_ads_service = get_google_ads_service(supabase)
        self.meta_ads_service = get_meta_ads_service(supabase)

        logger.info("Ads Ingestion Service initialized")

    def _run_async(self, coro):
        """
        Helper to run async coroutine from sync context.

        Args:
            coro: Coroutine to run

        Returns:
            Result of the coroutine
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, schedule as task
                asyncio.create_task(coro)
            else:
                # If no event loop, run directly
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop exists, create one
            return asyncio.run(coro)

    def connect_google_ads_account(
        self,
        org_id: str,
        access_token: str,
        refresh_token: str,
        connection_id: str
    ) -> List[Dict[str, Any]]:
        """
        Discover and store Google Ads accounts for an organization.

        Args:
            org_id: Organization ID
            access_token: OAuth access token
            refresh_token: OAuth refresh token
            connection_id: Connection ID from connections table

        Returns:
            List of created ad account records
        """
        logger.info(f"[GOOGLE_ADS] Connecting accounts for org {org_id}")

        try:
            # Fetch accessible accounts from Google Ads API
            accounts = self.google_ads_service.list_accessible_customers(access_token)

            created_accounts = []

            for account in accounts:
                # Check if account already exists
                existing = self.supabase.table("ad_accounts")\
                    .select("*")\
                    .eq("org_id", org_id)\
                    .eq("platform", "google_ads")\
                    .eq("account_id", account["customer_id"])\
                    .execute()

                if existing.data:
                    # Update existing account
                    updated = self.supabase.table("ad_accounts")\
                        .update({
                            "account_name": account["descriptive_name"],
                            "currency": account["currency_code"],
                            "timezone": account["time_zone"],
                            "status": "active" if account["status"] == "ENABLED" else "suspended",
                            "connection_id": connection_id,
                            "updated_at": datetime.utcnow().isoformat()
                        })\
                        .eq("id", existing.data[0]["id"])\
                        .execute()

                    created_accounts.append(updated.data[0])
                    logger.info(f"[GOOGLE_ADS] Updated account {account['customer_id']}")
                else:
                    # Insert new account
                    new_account = self.supabase.table("ad_accounts").insert({
                        "org_id": org_id,
                        "platform": "google_ads",
                        "account_id": account["customer_id"],
                        "account_name": account["descriptive_name"],
                        "currency": account["currency_code"],
                        "timezone": account["time_zone"],
                        "status": "active" if account["status"] == "ENABLED" else "suspended",
                        "connection_id": connection_id,
                        "metadata": {
                            "manager": account.get("manager", False),
                            "test_account": account.get("test_account", False),
                            "can_manage_clients": account.get("can_manage_clients", False)
                        }
                    }).execute()

                    created_accounts.append(new_account.data[0])
                    logger.info(f"[GOOGLE_ADS] Created account {account['customer_id']}")

            logger.info(f"[GOOGLE_ADS] Connected {len(created_accounts)} accounts")
            return created_accounts

        except Exception as e:
            logger.error(f"[GOOGLE_ADS] Failed to connect accounts: {e}", exc_info=True)
            raise

    def connect_meta_ads_account(
        self,
        org_id: str,
        access_token: str,
        connection_id: str
    ) -> List[Dict[str, Any]]:
        """
        Discover and store Meta Ads accounts for an organization.

        Args:
            org_id: Organization ID
            access_token: OAuth access token
            connection_id: Connection ID from connections table

        Returns:
            List of created ad account records
        """
        logger.info(f"[META_ADS] Connecting accounts for org {org_id}")

        try:
            # Fetch accessible ad accounts from Meta Marketing API
            accounts = self.meta_ads_service.list_ad_accounts(access_token)

            created_accounts = []

            for account in accounts:
                # Check if account already exists
                existing = self.supabase.table("ad_accounts")\
                    .select("*")\
                    .eq("org_id", org_id)\
                    .eq("platform", "meta_ads")\
                    .eq("account_id", account["account_id"])\
                    .execute()

                if existing.data:
                    # Update existing account
                    updated = self.supabase.table("ad_accounts")\
                        .update({
                            "account_name": account["name"],
                            "currency": account["currency"],
                            "timezone": account["timezone_name"],
                            "status": "active" if account["account_status"] == 1 else "inactive",
                            "connection_id": connection_id,
                            "updated_at": datetime.utcnow().isoformat()
                        })\
                        .eq("id", existing.data[0]["id"])\
                        .execute()

                    created_accounts.append(updated.data[0])
                    logger.info(f"[META_ADS] Updated account {account['account_id']}")
                else:
                    # Insert new account
                    new_account = self.supabase.table("ad_accounts").insert({
                        "org_id": org_id,
                        "platform": "meta_ads",
                        "account_id": account["account_id"],
                        "account_name": account["name"],
                        "currency": account["currency"],
                        "timezone": account["timezone_name"],
                        "status": "active" if account["account_status"] == 1 else "inactive",
                        "connection_id": connection_id,
                        "metadata": {
                            "business_name": account.get("business_name"),
                            "business_id": account.get("business_id"),
                            "is_prepay": account.get("is_prepay_account", False)
                        }
                    }).execute()

                    created_accounts.append(new_account.data[0])
                    logger.info(f"[META_ADS] Created account {account['account_id']}")

            logger.info(f"[META_ADS] Connected {len(created_accounts)} accounts")
            return created_accounts

        except Exception as e:
            logger.error(f"[META_ADS] Failed to connect accounts: {e}", exc_info=True)
            raise

    def sync_account_data(
        self,
        account_id: str,
        org_id: str,
        days_back: int = 90,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronize campaigns and metrics for an ad account.

        Args:
            account_id: Ad account UUID from database
            org_id: Organization ID
            days_back: Number of days of historical data to fetch
            force_refresh: If True, re-fetch all data even if synced recently

        Returns:
            Sync job summary
        """
        start_time = time.time()

        # Get account details
        account = self.supabase.table("ad_accounts")\
            .select("*")\
            .eq("id", account_id)\
            .eq("org_id", org_id)\
            .single()\
            .execute()

        if not account.data:
            raise ValueError(f"Ad account {account_id} not found for org {org_id}")

        account_data = account.data
        platform = account_data["platform"]
        external_account_id = account_data["account_id"]

        logger.info(
            f"[{platform.upper()}] Starting sync for account {external_account_id} "
            f"(org: {org_id}, days_back: {days_back})"
        )

        # Create sync job record
        job_id = str(uuid.uuid4())
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        self.supabase.table("ad_sync_jobs").insert({
            "id": job_id,
            "org_id": org_id,
            "account_id": account_id,
            "platform": platform,
            "sync_type": "initial" if force_refresh else "scheduled",
            "status": "running",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "started_at": datetime.utcnow().isoformat()
        }).execute()

        campaigns_synced = 0
        metrics_synced = 0
        errors = []

        try:
            # Get access token from connection (or use mock token if no connection)
            if account_data.get("connection_id"):
                # Production mode: read from connections table
                connection = self.supabase.table("connections")\
                    .select("access_token")\
                    .eq("id", account_data["connection_id"])\
                    .single()\
                    .execute()

                access_token = connection.data["access_token"]
            else:
                # Mock/simulated mode: use dummy token
                logger.info(f"[MOCK] No connection_id, using mock access token")
                access_token = "mock_access_token"

            # Sync campaigns
            if platform == "google_ads":
                campaigns_synced = self._sync_google_ads_campaigns(
                    org_id, account_id, external_account_id, access_token
                )
                metrics_synced = self._sync_google_ads_metrics(
                    org_id, account_id, external_account_id, access_token,
                    start_date, end_date
                )
            elif platform == "meta_ads":
                campaigns_synced = self._sync_meta_ads_campaigns(
                    org_id, account_id, external_account_id, access_token
                )
                metrics_synced = self._sync_meta_ads_metrics(
                    org_id, account_id, external_account_id, access_token,
                    start_date, end_date
                )

            # Update account last sync timestamp
            self.supabase.table("ad_accounts")\
                .update({"last_sync_at": datetime.utcnow().isoformat()})\
                .eq("id", account_id)\
                .execute()

            # Mark job as completed
            duration_ms = int((time.time() - start_time) * 1000)
            self.supabase.table("ad_sync_jobs")\
                .update({
                    "status": "completed",
                    "campaigns_synced": campaigns_synced,
                    "metrics_synced": metrics_synced,
                    "errors_count": len(errors),
                    "error_details": {"errors": errors} if errors else None,
                    "completed_at": datetime.utcnow().isoformat(),
                    "duration_ms": duration_ms
                })\
                .eq("id", job_id)\
                .execute()

            logger.info(
                f"[{platform.upper()}] Sync completed: "
                f"{campaigns_synced} campaigns, {metrics_synced} metrics in {duration_ms}ms"
            )

            return {
                "job_id": job_id,
                "status": "completed",
                "campaigns_synced": campaigns_synced,
                "metrics_synced": metrics_synced,
                "duration_ms": duration_ms,
                "errors_count": len(errors)
            }

        except Exception as e:
            # Mark job as failed
            duration_ms = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            errors.append(error_msg)

            self.supabase.table("ad_sync_jobs")\
                .update({
                    "status": "failed",
                    "campaigns_synced": campaigns_synced,
                    "metrics_synced": metrics_synced,
                    "errors_count": len(errors),
                    "error_details": {"errors": errors},
                    "completed_at": datetime.utcnow().isoformat(),
                    "duration_ms": duration_ms
                })\
                .eq("id", job_id)\
                .execute()

            logger.error(f"[{platform.upper()}] Sync failed: {error_msg}", exc_info=True)
            raise

    def _sync_google_ads_campaigns(
        self,
        org_id: str,
        account_id: str,
        customer_id: str,
        access_token: str
    ) -> int:
        """Sync Google Ads campaigns for an account."""
        campaigns = self.google_ads_service.get_campaigns(access_token, customer_id)

        synced_count = 0

        for campaign in campaigns:
            try:
                # Check if campaign exists
                existing = self.supabase.table("ad_campaigns")\
                    .select("id")\
                    .eq("account_id", account_id)\
                    .eq("campaign_id", campaign["id"])\
                    .execute()

                campaign_data = {
                    "org_id": org_id,
                    "account_id": account_id,
                    "platform": "google_ads",
                    "campaign_id": campaign["id"],
                    "campaign_name": campaign["name"],
                    "objective": campaign.get("advertising_channel_type", "").lower(),
                    "status": campaign["status"].lower(),
                    "budget_amount": campaign["budget"]["amount_micros"] / 1_000_000 if campaign.get("budget") else None,
                    "budget_type": "daily",  # Google Ads typically uses daily budgets
                    "bid_strategy": campaign.get("bidding_strategy_type", "").lower(),
                    "start_date": campaign.get("start_date"),
                    "end_date": campaign.get("end_date"),
                    "targeting": campaign.get("targeting_setting", {}),
                    "google_ads_data": {
                        "network_settings": campaign.get("network_settings", {}),
                        "ad_rotation_mode": campaign.get("ad_rotation_mode", "").lower()
                    },
                    "last_sync_at": datetime.utcnow().isoformat()
                }

                if existing.data:
                    # Update
                    self.supabase.table("ad_campaigns")\
                        .update(campaign_data)\
                        .eq("id", existing.data[0]["id"])\
                        .execute()
                    campaign_db_id = existing.data[0]["id"]
                else:
                    # Insert
                    result = self.supabase.table("ad_campaigns").insert(campaign_data).execute()
                    campaign_db_id = result.data[0]["id"]

                # Create document and embedding for semantic search
                try:
                    self._create_campaign_document(campaign_db_id, org_id, campaign_data)
                except Exception as doc_error:
                    logger.error(f"[GOOGLE_ADS] Failed to create document for campaign {campaign_db_id}: {doc_error}")

                # Create memory for campaign event
                try:
                    # Get latest metrics for memory context
                    latest_metrics = self.supabase.table("ad_metrics")\
                        .select("*")\
                        .eq("campaign_id", campaign_db_id)\
                        .order("metric_date", desc=True)\
                        .limit(1)\
                        .execute()

                    metrics = latest_metrics.data[0] if latest_metrics.data else None
                    is_new_campaign = not bool(existing.data)

                    self._run_async(
                        self._create_campaign_memory(
                            campaign_db_id=campaign_db_id,
                            org_id=org_id,
                            campaign_data=campaign_data,
                            metrics=metrics,
                            is_new=is_new_campaign
                        )
                    )
                except Exception as mem_error:
                    logger.error(f"[GOOGLE_ADS] Failed to create memory for campaign {campaign_db_id}: {mem_error}")

                synced_count += 1

            except Exception as e:
                logger.error(f"[GOOGLE_ADS] Failed to sync campaign {campaign.get('id')}: {e}")

        return synced_count

    def _sync_google_ads_metrics(
        self,
        org_id: str,
        account_id: str,
        customer_id: str,
        access_token: str,
        start_date: date,
        end_date: date
    ) -> int:
        """Sync Google Ads metrics for all campaigns."""
        # Get all campaigns for this account
        campaigns = self.supabase.table("ad_campaigns")\
            .select("id, campaign_id")\
            .eq("account_id", account_id)\
            .eq("platform", "google_ads")\
            .execute()

        synced_count = 0

        for campaign in campaigns.data:
            try:
                # Fetch metrics from API
                metrics = self.google_ads_service.get_campaign_metrics(
                    access_token, customer_id, campaign["campaign_id"],
                    start_date, end_date
                )

                # Upsert each day's metrics
                for metric in metrics:
                    metric_data = {
                        "campaign_id": campaign["id"],
                        "metric_date": metric["metric_date"],
                        **{k: v for k, v in metric.items() if k != "metric_date"}
                    }

                    # Use upsert (insert or update if exists)
                    self.supabase.table("ad_metrics")\
                        .upsert(metric_data, on_conflict="campaign_id,metric_date")\
                        .execute()

                    synced_count += 1

            except Exception as e:
                logger.error(f"[GOOGLE_ADS] Failed to sync metrics for campaign {campaign['campaign_id']}: {e}")

        return synced_count

    def _sync_meta_ads_campaigns(
        self,
        org_id: str,
        account_id: str,
        ad_account_id: str,
        access_token: str
    ) -> int:
        """Sync Meta Ads campaigns for an account."""
        campaigns = self.meta_ads_service.get_campaigns(access_token, ad_account_id)

        synced_count = 0

        for campaign in campaigns:
            try:
                # Check if campaign exists
                existing = self.supabase.table("ad_campaigns")\
                    .select("id")\
                    .eq("account_id", account_id)\
                    .eq("campaign_id", campaign["id"])\
                    .execute()

                # Parse dates
                start_date = campaign.get("start_time", "").split("T")[0] if campaign.get("start_time") else None
                end_date = campaign.get("stop_time", "").split("T")[0] if campaign.get("stop_time") else None

                # Determine budget
                budget_amount = None
                budget_type = None
                if campaign.get("daily_budget"):
                    budget_amount = campaign["daily_budget"] / 100
                    budget_type = "daily"
                elif campaign.get("lifetime_budget"):
                    budget_amount = campaign["lifetime_budget"] / 100
                    budget_type = "lifetime"

                campaign_data = {
                    "org_id": org_id,
                    "account_id": account_id,
                    "platform": "meta_ads",
                    "campaign_id": campaign["id"],
                    "campaign_name": campaign["name"],
                    "objective": campaign.get("objective", "").lower(),
                    "status": campaign["status"].lower(),
                    "budget_amount": budget_amount,
                    "budget_type": budget_type,
                    "bid_strategy": campaign.get("bid_strategy", "").lower(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "meta_ads_data": {
                        "buying_type": campaign.get("buying_type", ""),
                        "special_ad_categories": campaign.get("special_ad_categories", []),
                        "created_time": campaign.get("created_time", ""),
                        "updated_time": campaign.get("updated_time", "")
                    },
                    "last_sync_at": datetime.utcnow().isoformat()
                }

                if existing.data:
                    # Update
                    self.supabase.table("ad_campaigns")\
                        .update(campaign_data)\
                        .eq("id", existing.data[0]["id"])\
                        .execute()
                    campaign_db_id = existing.data[0]["id"]
                else:
                    # Insert
                    result = self.supabase.table("ad_campaigns").insert(campaign_data).execute()
                    campaign_db_id = result.data[0]["id"]

                # Create document and embedding for semantic search
                try:
                    self._create_campaign_document(campaign_db_id, org_id, campaign_data)
                except Exception as doc_error:
                    logger.error(f"[META_ADS] Failed to create document for campaign {campaign_db_id}: {doc_error}")

                # Create memory for campaign event
                try:
                    # Get latest metrics for memory context
                    latest_metrics = self.supabase.table("ad_metrics")\
                        .select("*")\
                        .eq("campaign_id", campaign_db_id)\
                        .order("metric_date", desc=True)\
                        .limit(1)\
                        .execute()

                    metrics = latest_metrics.data[0] if latest_metrics.data else None
                    is_new_campaign = not bool(existing.data)

                    self._run_async(
                        self._create_campaign_memory(
                            campaign_db_id=campaign_db_id,
                            org_id=org_id,
                            campaign_data=campaign_data,
                            metrics=metrics,
                            is_new=is_new_campaign
                        )
                    )
                except Exception as mem_error:
                    logger.error(f"[META_ADS] Failed to create memory for campaign {campaign_db_id}: {mem_error}")

                synced_count += 1

            except Exception as e:
                logger.error(f"[META_ADS] Failed to sync campaign {campaign.get('id')}: {e}")

        return synced_count

    def _sync_meta_ads_metrics(
        self,
        org_id: str,
        account_id: str,
        ad_account_id: str,
        access_token: str,
        start_date: date,
        end_date: date
    ) -> int:
        """Sync Meta Ads insights for all campaigns."""
        # Get all campaigns for this account
        campaigns = self.supabase.table("ad_campaigns")\
            .select("id, campaign_id")\
            .eq("account_id", account_id)\
            .eq("platform", "meta_ads")\
            .execute()

        synced_count = 0

        for campaign in campaigns.data:
            try:
                # Fetch insights from API
                insights = self.meta_ads_service.get_campaign_insights(
                    access_token, campaign["campaign_id"],
                    start_date, end_date
                )

                # Parse and upsert each day's insights
                for insight in insights:
                    # Extract conversions from actions
                    conversions = 0
                    conversion_value = 0
                    engagement_count = 0

                    for action in insight.get("actions", []):
                        if action["action_type"] == "offsite_conversion":
                            conversions = int(action["value"])
                        elif action["action_type"] == "post_engagement":
                            engagement_count = int(action["value"])

                    for value in insight.get("action_values", []):
                        if value["action_type"] == "offsite_conversion":
                            conversion_value = float(value["value"])

                    # Calculate derived metrics
                    impressions = int(insight["impressions"])
                    clicks = int(insight["clicks"])
                    spend = float(insight["spend"])

                    conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0
                    cpa = (spend / conversions) if conversions > 0 else None
                    roas = (conversion_value / spend) if spend > 0 else None
                    engagement_rate = (engagement_count / impressions * 100) if impressions > 0 else None

                    metric_data = {
                        "campaign_id": campaign["id"],
                        "metric_date": insight["date_start"],
                        "impressions": impressions,
                        "clicks": clicks,
                        "ctr": float(insight["ctr"]),
                        "spend": spend,
                        "conversions": conversions,
                        "conversion_rate": round(conversion_rate, 2) if conversion_rate > 0 else None,
                        "cpa": round(cpa, 2) if cpa else None,
                        "roas": round(roas, 2) if roas else None,
                        "conversion_value": round(conversion_value, 2) if conversion_value > 0 else None,
                        "engagement_rate": round(engagement_rate, 2) if engagement_rate else None,
                        "engagement_count": engagement_count if engagement_count > 0 else None,
                        "reach": int(insight.get("reach", 0)),
                        "frequency": float(insight.get("frequency", 1)),
                        "cpc": float(insight["cpc"]),
                        "cpm": float(insight["cpm"]),
                        "meta_ads_metrics": {
                            "actions": insight.get("actions", []),
                            "action_values": insight.get("action_values", []),
                            "video_30_sec_watched": insight.get("video_30_sec_watched_actions", [])
                        }
                    }

                    # Upsert
                    self.supabase.table("ad_metrics")\
                        .upsert(metric_data, on_conflict="campaign_id,metric_date")\
                        .execute()

                    synced_count += 1

            except Exception as e:
                logger.error(f"[META_ADS] Failed to sync metrics for campaign {campaign['campaign_id']}: {e}")

        return synced_count

    async def _create_campaign_memory(
        self,
        campaign_db_id: str,
        org_id: str,
        campaign_data: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        is_new: bool = False
    ) -> None:
        """
        Create memory records for campaign events and insights.

        Args:
            campaign_db_id: Database ID of the campaign
            org_id: Organization ID
            campaign_data: Campaign data dict
            metrics: Latest metrics for the campaign
            is_new: Whether this is a newly created campaign
        """
        try:
            memory_service = get_memory_service(self.supabase)
            platform = campaign_data.get("platform", "unknown")
            campaign_name = campaign_data.get("campaign_name", "Unnamed Campaign")
            status = campaign_data.get("status", "unknown")

            # Determine memory type and importance based on event
            if is_new:
                # New campaign launch - store as "update" memory
                title = f"New {platform.replace('_', ' ').title()} Campaign: {campaign_name}"
                content = f"Campaign '{campaign_name}' launched on {platform.replace('_', ' ').title()} with status: {status}"

                if campaign_data.get("objective"):
                    content += f"\nObjective: {campaign_data['objective']}"
                if campaign_data.get("budget_amount"):
                    content += f"\nBudget: {campaign_data['budget_amount']} {campaign_data.get('currency', 'USD')}"

                await memory_service.store_memory(
                    org_id=org_id,
                    title=title,
                    content=content,
                    memory_type="update",
                    importance=0.6,
                    source_refs=[{"campaign_id": campaign_db_id, "platform": platform}],
                    metadata={
                        "campaign_id": campaign_db_id,
                        "platform": platform,
                        "event_type": "campaign_launch"
                    }
                )
                logger.info(f"[MEMORY] Created launch memory for campaign {campaign_name}")

            # Check for performance milestones
            if metrics:
                roas = float(metrics.get("roas", 0) or 0)
                conversions = int(metrics.get("conversions", 0) or 0)

                # High performance - create insight memory
                # Adjusted thresholds: ROAS >= 4.0 OR (ROAS >= 3.0 AND conversions >= 10)
                is_high_performer = (roas >= 4.0) or (roas >= 3.0 and conversions >= 10)

                if is_high_performer:
                    title = f"High-Performing Campaign: {campaign_name}"
                    content = f"Campaign '{campaign_name}' on {platform.replace('_', ' ').title()} is performing exceptionally well:\n"
                    content += f"- ROAS: {roas:.2f}x\n"
                    content += f"- Conversions: {conversions}\n"
                    content += f"- CTR: {metrics.get('ctr', 0):.2f}%\n"
                    content += f"\nThis campaign could be worth replicating or scaling up budget."

                    await memory_service.store_memory(
                        org_id=org_id,
                        title=title,
                        content=content,
                        memory_type="insight",
                        importance=0.9,  # High importance for high performers
                        source_refs=[{"campaign_id": campaign_db_id, "platform": platform}],
                        metadata={
                            "campaign_id": campaign_db_id,
                            "platform": platform,
                            "event_type": "high_performance",
                            "roas": roas,
                            "conversions": conversions
                        }
                    )
                    logger.info(f"[MEMORY] Created high-performance insight for campaign {campaign_name}")

                # Poor performance - create insight memory
                elif roas < 1.0 and conversions < 10:
                    title = f"Underperforming Campaign Alert: {campaign_name}"
                    content = f"Campaign '{campaign_name}' on {platform.replace('_', ' ').title()} is underperforming:\n"
                    content += f"- ROAS: {roas:.2f}x (below 1.0 - losing money)\n"
                    content += f"- Conversions: {conversions}\n"
                    content += f"- CTR: {metrics.get('ctr', 0):.2f}%\n"
                    content += f"\nConsider pausing this campaign or adjusting targeting/creative."

                    await memory_service.store_memory(
                        org_id=org_id,
                        title=title,
                        content=content,
                        memory_type="insight",
                        importance=0.8,  # High importance for issues
                        source_refs=[{"campaign_id": campaign_db_id, "platform": platform}],
                        metadata={
                            "campaign_id": campaign_db_id,
                            "platform": platform,
                            "event_type": "poor_performance",
                            "roas": roas,
                            "conversions": conversions
                        }
                    )
                    logger.info(f"[MEMORY] Created underperformance alert for campaign {campaign_name}")

            # Status change - create update memory
            if status in ["paused", "ended", "deleted"]:
                title = f"Campaign Status Change: {campaign_name}"
                content = f"Campaign '{campaign_name}' on {platform.replace('_', ' ').title()} status changed to: {status}"

                if metrics:
                    content += f"\n\nFinal Performance:"
                    content += f"\n- ROAS: {metrics.get('roas', 0):.2f}x"
                    content += f"\n- Total Conversions: {metrics.get('conversions', 0)}"
                    content += f"\n- Total Spend: {metrics.get('spend', 0):.2f}"

                await memory_service.store_memory(
                    org_id=org_id,
                    title=title,
                    content=content,
                    memory_type="update",
                    importance=0.5,
                    source_refs=[{"campaign_id": campaign_db_id, "platform": platform}],
                    metadata={
                        "campaign_id": campaign_db_id,
                        "platform": platform,
                        "event_type": "status_change",
                        "new_status": status
                    }
                )
                logger.info(f"[MEMORY] Created status change memory for campaign {campaign_name}")

        except Exception as e:
            logger.error(f"Failed to create campaign memory: {e}", exc_info=True)
            # Don't raise - memory creation failures shouldn't block campaign ingestion

    def _create_campaign_document(
        self,
        campaign_db_id: str,
        org_id: str,
        campaign_data: Dict[str, Any]
    ) -> None:
        """
        Create a searchable document and embedding for a campaign.

        Args:
            campaign_db_id: Database ID of the campaign (UUID)
            org_id: Organization ID
            campaign_data: Campaign data dict
        """
        try:
            # Get latest metrics for this campaign
            latest_metrics = self.supabase.table("ad_metrics")\
                .select("*")\
                .eq("campaign_id", campaign_db_id)\
                .order("metric_date", desc=True)\
                .limit(1)\
                .execute()

            metrics = latest_metrics.data[0] if latest_metrics.data else None

            # Convert campaign to document format
            doc_service = get_ads_document_service()
            doc_content = doc_service.campaign_to_document(campaign_data, metrics)

            platform = campaign_data["platform"]
            campaign_id = campaign_data["campaign_id"]

            # Check if document already exists for this campaign
            existing_doc = self.supabase.table("documents")\
                .select("id")\
                .eq("org_id", org_id)\
                .eq("source_type", platform)\
                .eq("source_id", campaign_id)\
                .execute()

            document_record = {
                "org_id": org_id,
                "source_type": platform,
                "source_id": campaign_id,
                "title": doc_content["title"],
                "content": doc_content["content"],
                "summary": doc_content["summary"],
                "metadata": {
                    "campaign_id": campaign_db_id,
                    "campaign_name": campaign_data["campaign_name"],
                    "platform": platform,
                    "status": campaign_data["status"],
                    "objective": campaign_data.get("objective"),
                    "type": "ad_campaign"
                },
                "embedding_status": "pending",
                "updated_at": datetime.utcnow().isoformat()
            }

            if existing_doc.data:
                # Update existing document
                self.supabase.table("documents")\
                    .update(document_record)\
                    .eq("id", existing_doc.data[0]["id"])\
                    .execute()
                document_id = existing_doc.data[0]["id"]
                logger.info(f"[{platform.upper()}] Updated document for campaign {campaign_id}")
            else:
                # Create new document
                result = self.supabase.table("documents")\
                    .insert(document_record)\
                    .execute()
                document_id = result.data[0]["id"]
                logger.info(f"[{platform.upper()}] Created document for campaign {campaign_id}")

            # Generate embedding
            embedding_service = get_embedding_service()
            embedding_vector = embedding_service.generate_embedding(doc_content["content"])

            # Check if embedding already exists
            existing_embedding = self.supabase.table("embeddings")\
                .select("id")\
                .eq("document_id", document_id)\
                .execute()

            embedding_record = {
                "org_id": org_id,
                "document_id": document_id,
                "content": doc_content["content"],
                "embedding": embedding_vector,
                "metadata": {
                    "source": "ad_campaign",
                    "platform": platform,
                    "campaign_id": campaign_db_id
                }
            }

            if existing_embedding.data:
                # Update existing embedding
                self.supabase.table("embeddings")\
                    .update(embedding_record)\
                    .eq("id", existing_embedding.data[0]["id"])\
                    .execute()
                logger.info(f"[{platform.upper()}] Updated embedding for campaign {campaign_id}")
            else:
                # Create new embedding
                self.supabase.table("embeddings")\
                    .insert(embedding_record)\
                    .execute()
                logger.info(f"[{platform.upper()}] Created embedding for campaign {campaign_id}")

            # Mark document as completed
            self.supabase.table("documents")\
                .update({"embedding_status": "completed"})\
                .eq("id", document_id)\
                .execute()

        except Exception as e:
            logger.error(f"Failed to create campaign document: {e}", exc_info=True)
            raise


# Singleton instance
_ads_ingestion_service: Optional[AdsIngestionService] = None


def get_ads_ingestion_service(supabase: Client) -> AdsIngestionService:
    """Get or create singleton ads ingestion service instance."""
    global _ads_ingestion_service
    if _ads_ingestion_service is None:
        _ads_ingestion_service = AdsIngestionService(supabase)
    return _ads_ingestion_service
