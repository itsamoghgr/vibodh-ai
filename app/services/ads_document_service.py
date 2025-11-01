# -*- coding: utf-8 -*-
"""
Ads Document Service
Converts ad campaign data into searchable document text for pgvector embeddings
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json


class AdsDocumentService:
    """Service for converting ad campaigns into searchable documents"""

    def __init__(self):
        pass

    def campaign_to_document(
        self,
        campaign: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Convert campaign data to document format for embedding.

        Creates rich, searchable text content from campaign data including:
        - Campaign name, objective, status
        - Budget and bid strategy
        - Targeting information
        - Performance metrics (if provided)

        Args:
            campaign: Campaign data from ad_campaigns table
            metrics: Optional latest metrics from ad_metrics table

        Returns:
            Dict with 'title', 'content', and 'summary' for document
        """
        # Extract campaign fields
        campaign_name = campaign.get('campaign_name', 'Unnamed Campaign')
        platform = campaign.get('platform', 'unknown')
        platform_display = "Google Ads" if platform == "google_ads" else "Meta Ads" if platform == "meta_ads" else platform

        objective = campaign.get('objective', 'Not specified')
        status = campaign.get('status', 'unknown')
        budget_amount = campaign.get('budget_amount')
        budget_type = campaign.get('budget_type')
        bid_strategy = campaign.get('bid_strategy')

        # Format dates
        start_date = campaign.get('start_date')
        end_date = campaign.get('end_date')

        # Build targeting summary
        targeting = campaign.get('targeting', {})
        targeting_summary = self._format_targeting(targeting)

        # Build title
        title = f"{platform_display} Campaign: {campaign_name}"

        # Build main content
        content_parts = [
            f"# {campaign_name}",
            f"\n**Platform:** {platform_display}",
            f"**Status:** {status.upper()}",
            f"**Objective:** {objective}",
        ]

        # Add budget information
        if budget_amount:
            budget_str = f"${budget_amount:,.2f}"
            if budget_type:
                budget_str += f" ({budget_type})"
            content_parts.append(f"**Budget:** {budget_str}")

        if bid_strategy:
            content_parts.append(f"**Bid Strategy:** {bid_strategy}")

        # Add date range
        if start_date or end_date:
            date_range = []
            if start_date:
                date_range.append(f"Start: {start_date}")
            if end_date:
                date_range.append(f"End: {end_date}")
            content_parts.append(f"**Schedule:** {', '.join(date_range)}")

        # Add targeting
        if targeting_summary:
            content_parts.append(f"\n## Targeting\n{targeting_summary}")

        # Add performance metrics if provided
        if metrics:
            metrics_summary = self._format_metrics(metrics)
            if metrics_summary:
                content_parts.append(f"\n## Performance Metrics\n{metrics_summary}")

        # Add platform-specific data
        if platform == 'google_ads' and campaign.get('google_ads_data'):
            google_data = self._format_platform_data(campaign['google_ads_data'], 'Google Ads')
            if google_data:
                content_parts.append(f"\n## Google Ads Details\n{google_data}")

        if platform == 'meta_ads' and campaign.get('meta_ads_data'):
            meta_data = self._format_platform_data(campaign['meta_ads_data'], 'Meta Ads')
            if meta_data:
                content_parts.append(f"\n## Meta Ads Details\n{meta_data}")

        # Combine all parts
        content = "\n".join(content_parts)

        # Create short summary for quick reference
        summary_parts = [
            f"{platform_display} campaign '{campaign_name}'",
            f"Status: {status}",
        ]

        if objective and objective != 'Not specified':
            summary_parts.append(f"Objective: {objective}")

        if metrics:
            roas = metrics.get('roas')
            conversions = metrics.get('conversions')
            if roas:
                summary_parts.append(f"ROAS: {roas:.2f}x")
            if conversions:
                summary_parts.append(f"Conversions: {conversions}")

        summary = ". ".join(summary_parts) + "."

        return {
            "title": title,
            "content": content,
            "summary": summary
        }

    def _format_targeting(self, targeting: Dict[str, Any]) -> str:
        """Format targeting data into readable text"""
        if not targeting or targeting == {}:
            return "No targeting information available."

        parts = []

        # Location targeting
        if targeting.get('locations'):
            locations = targeting['locations']
            if isinstance(locations, list):
                parts.append(f"- **Locations:** {', '.join(locations)}")
            else:
                parts.append(f"- **Locations:** {locations}")

        # Age targeting
        if targeting.get('age_min') or targeting.get('age_max'):
            age_min = targeting.get('age_min', '18')
            age_max = targeting.get('age_max', '65+')
            parts.append(f"- **Age Range:** {age_min} to {age_max}")

        # Gender targeting
        if targeting.get('genders'):
            parts.append(f"- **Gender:** {targeting['genders']}")

        # Interests
        if targeting.get('interests'):
            interests = targeting['interests']
            if isinstance(interests, list):
                parts.append(f"- **Interests:** {', '.join(interests)}")
            else:
                parts.append(f"- **Interests:** {interests}")

        # Keywords (Google Ads)
        if targeting.get('keywords'):
            keywords = targeting['keywords']
            if isinstance(keywords, list):
                parts.append(f"- **Keywords:** {', '.join(keywords[:10])}")  # Limit to 10
            else:
                parts.append(f"- **Keywords:** {keywords}")

        # Audiences
        if targeting.get('audiences'):
            audiences = targeting['audiences']
            if isinstance(audiences, list):
                parts.append(f"- **Audiences:** {', '.join(audiences)}")
            else:
                parts.append(f"- **Audiences:** {audiences}")

        # Devices
        if targeting.get('devices'):
            parts.append(f"- **Devices:** {targeting['devices']}")

        # Placements (Meta Ads)
        if targeting.get('placements'):
            placements = targeting['placements']
            if isinstance(placements, list):
                parts.append(f"- **Placements:** {', '.join(placements)}")
            else:
                parts.append(f"- **Placements:** {placements}")

        return "\n".join(parts) if parts else "Broad targeting (no specific criteria set)"

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics into readable text"""
        parts = []

        # Key metrics
        impressions = metrics.get('impressions')
        clicks = metrics.get('clicks')
        ctr = metrics.get('ctr')
        spend = metrics.get('spend')
        conversions = metrics.get('conversions')
        conversion_rate = metrics.get('conversion_rate')
        roas = metrics.get('roas')
        cpa = metrics.get('cpa')

        if impressions is not None:
            parts.append(f"- **Impressions:** {impressions:,}")

        if clicks is not None:
            parts.append(f"- **Clicks:** {clicks:,}")

        if ctr is not None:
            parts.append(f"- **CTR:** {ctr:.2f}%")

        if spend is not None:
            parts.append(f"- **Spend:** ${spend:,.2f}")

        if conversions is not None:
            parts.append(f"- **Conversions:** {conversions:,}")

        if conversion_rate is not None:
            parts.append(f"- **Conversion Rate:** {conversion_rate:.2f}%")

        if roas is not None:
            parts.append(f"- **ROAS:** {roas:.2f}x")

        if cpa is not None:
            parts.append(f"- **CPA:** ${cpa:.2f}")

        # Additional metrics
        quality_score = metrics.get('quality_score')
        engagement_rate = metrics.get('engagement_rate')

        if quality_score is not None:
            parts.append(f"- **Quality Score:** {quality_score}/10")

        if engagement_rate is not None:
            parts.append(f"- **Engagement Rate:** {engagement_rate:.2f}%")

        # Metric date
        metric_date = metrics.get('metric_date')
        if metric_date:
            parts.append(f"\n*Latest metrics as of {metric_date}*")

        return "\n".join(parts) if parts else "No performance data available yet."

    def _format_platform_data(self, platform_data: Dict[str, Any], platform_name: str) -> str:
        """Format platform-specific data into readable text"""
        if not platform_data or platform_data == {}:
            return ""

        parts = []

        # Iterate through platform-specific fields
        for key, value in platform_data.items():
            # Skip empty values
            if value is None or value == "" or value == {}:
                continue

            # Format key as readable label
            label = key.replace('_', ' ').title()

            # Format value based on type
            if isinstance(value, dict):
                value_str = json.dumps(value, indent=2)
            elif isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)

            parts.append(f"- **{label}:** {value_str}")

        return "\n".join(parts) if parts else ""


def get_ads_document_service() -> AdsDocumentService:
    """Get singleton instance of AdsDocumentService"""
    return AdsDocumentService()
