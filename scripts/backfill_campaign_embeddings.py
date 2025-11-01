#!/usr/bin/env python3
"""
Backfill Campaign Embeddings Script

Creates documents and embeddings for existing ad campaigns that don't have them yet.
This script should be run once after enabling embedding support for ads.

Usage:
    python scripts/backfill_campaign_embeddings.py [--org-id ORG_ID]
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.db.supabase_client import get_supabase_admin_client
from app.core.logging import logger
from app.services.ads_document_service import get_ads_document_service
from app.services.embedding_service import get_embedding_service
from datetime import datetime
import argparse


def backfill_campaign_embeddings(org_id: str = None):
    """
    Create documents and embeddings for existing campaigns.

    Args:
        org_id: Optional organization ID to filter campaigns. If None, processes all orgs.
    """
    supabase = get_supabase_admin_client()
    doc_service = get_ads_document_service()
    embedding_service = get_embedding_service()

    logger.info("=" * 80)
    logger.info("Starting Campaign Embeddings Backfill")
    logger.info("=" * 80)

    # Build query
    query = supabase.table("ad_campaigns").select("*")
    if org_id:
        query = query.eq("org_id", org_id)
        logger.info(f"Filtering by org_id: {org_id}")

    # Fetch all campaigns
    campaigns_result = query.execute()
    campaigns = campaigns_result.data

    logger.info(f"Found {len(campaigns)} total campaigns")

    # Track statistics
    stats = {
        "total": len(campaigns),
        "processed": 0,
        "created": 0,
        "updated": 0,
        "skipped": 0,
        "errors": 0
    }

    for idx, campaign in enumerate(campaigns, 1):
        campaign_id = campaign["id"]
        campaign_name = campaign["campaign_name"]
        platform = campaign["platform"]
        external_campaign_id = campaign["campaign_id"]
        org_id_val = campaign["org_id"]

        logger.info(f"\n[{idx}/{len(campaigns)}] Processing: {campaign_name} ({platform})")

        try:
            # Check if document already exists
            existing_doc = supabase.table("documents")\
                .select("id, embedding_status")\
                .eq("org_id", org_id_val)\
                .eq("source_type", platform)\
                .eq("source_id", external_campaign_id)\
                .execute()

            if existing_doc.data and existing_doc.data[0].get("embedding_status") == "completed":
                logger.info(f"  ✓ Document already exists with embedding - skipping")
                stats["skipped"] += 1
                continue

            # Get latest metrics
            latest_metrics = supabase.table("ad_metrics")\
                .select("*")\
                .eq("campaign_id", campaign_id)\
                .order("metric_date", desc=True)\
                .limit(1)\
                .execute()

            metrics = latest_metrics.data[0] if latest_metrics.data else None

            # Convert to document
            doc_content = doc_service.campaign_to_document(campaign, metrics)

            # Prepare document record
            document_record = {
                "org_id": org_id_val,
                "source_type": platform,
                "source_id": external_campaign_id,
                "title": doc_content["title"],
                "content": doc_content["content"],
                "summary": doc_content["summary"],
                "metadata": {
                    "campaign_id": campaign_id,
                    "campaign_name": campaign_name,
                    "platform": platform,
                    "status": campaign["status"],
                    "objective": campaign.get("objective"),
                    "type": "ad_campaign"
                },
                "embedding_status": "pending",
                "updated_at": datetime.utcnow().isoformat()
            }

            # Create or update document
            if existing_doc.data:
                supabase.table("documents")\
                    .update(document_record)\
                    .eq("id", existing_doc.data[0]["id"])\
                    .execute()
                document_id = existing_doc.data[0]["id"]
                logger.info(f"  ✓ Updated document")
                stats["updated"] += 1
            else:
                result = supabase.table("documents")\
                    .insert(document_record)\
                    .execute()
                document_id = result.data[0]["id"]
                logger.info(f"  ✓ Created document")
                stats["created"] += 1

            # Generate embedding
            logger.info(f"  → Generating embedding...")
            embedding_vector = embedding_service.generate_embedding(doc_content["content"])

            # Check if embedding exists
            existing_embedding = supabase.table("embeddings")\
                .select("id")\
                .eq("document_id", document_id)\
                .execute()

            embedding_record = {
                "org_id": org_id_val,
                "document_id": document_id,
                "content": doc_content["content"],
                "embedding": embedding_vector,
                "metadata": {
                    "source": "ad_campaign",
                    "platform": platform,
                    "campaign_id": campaign_id
                }
            }

            if existing_embedding.data:
                supabase.table("embeddings")\
                    .update(embedding_record)\
                    .eq("id", existing_embedding.data[0]["id"])\
                    .execute()
                logger.info(f"  ✓ Updated embedding")
            else:
                supabase.table("embeddings")\
                    .insert(embedding_record)\
                    .execute()
                logger.info(f"  ✓ Created embedding")

            # Mark document as completed
            supabase.table("documents")\
                .update({"embedding_status": "completed"})\
                .eq("id", document_id)\
                .execute()

            stats["processed"] += 1
            logger.info(f"  ✓ SUCCESS: Campaign embedded successfully")

        except Exception as e:
            logger.error(f"  ✗ ERROR: Failed to process campaign: {e}")
            stats["errors"] += 1

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Backfill Summary")
    logger.info("=" * 80)
    logger.info(f"Total campaigns: {stats['total']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"  - Created new: {stats['created']}")
    logger.info(f"  - Updated existing: {stats['updated']}")
    logger.info(f"Skipped (already embedded): {stats['skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info("=" * 80)

    if stats["errors"] > 0:
        logger.warning(f"⚠️  Backfill completed with {stats['errors']} errors")
        return 1
    else:
        logger.info("✅ Backfill completed successfully!")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for existing ad campaigns"
    )
    parser.add_argument(
        "--org-id",
        type=str,
        help="Organization ID to filter campaigns (optional, processes all if not specified)",
        default=None
    )

    args = parser.parse_args()

    try:
        exit_code = backfill_campaign_embeddings(org_id=args.org_id)
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Backfill failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
