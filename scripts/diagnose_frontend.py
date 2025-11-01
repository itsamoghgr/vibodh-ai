#!/usr/bin/env python3
"""
Diagnose why ads aren't showing in frontend
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.supabase_client import get_supabase_admin_client
from app.core.logging import logger


def diagnose():
    supabase = get_supabase_admin_client()

    print("=" * 80)
    print("ADS DISPLAY DIAGNOSTICS")
    print("=" * 80)
    print()

    # 1. Check documents by org and source type
    print("1. DOCUMENTS BY ORG AND SOURCE TYPE (ads only)")
    print("-" * 50)
    ads_docs = supabase.table("documents")\
        .select("org_id, source_type, embedding_status")\
        .in_("source_type", ["google_ads", "meta_ads"])\
        .execute()

    if ads_docs.data:
        # Group manually
        from collections import defaultdict
        org_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'completed': 0}))
        for doc in ads_docs.data:
            org_stats[doc['org_id']][doc['source_type']]['total'] += 1
            if doc['embedding_status'] == 'completed':
                org_stats[doc['org_id']][doc['source_type']]['completed'] += 1

        for org_id, sources in org_stats.items():
            print(f"  Org: {org_id}")
            for source, stats in sources.items():
                print(f"    {source}: {stats['total']} docs ({stats['completed']} completed)")
    else:
        print("  ❌ No ads documents found!")
    print()

    # 2. All document counts
    print("2. ALL DOCUMENT COUNTS BY SOURCE TYPE")
    print("-" * 50)
    all_docs = supabase.table("documents")\
        .select("source_type", count="exact")\
        .execute()

    # Group by source type manually
    from collections import Counter
    if all_docs.data:
        source_counts = Counter([doc['source_type'] for doc in all_docs.data])
        for source, count in source_counts.most_common():
            print(f"  {source}: {count}")
    print()

    # 3. KG entities by type
    print("3. KNOWLEDGE GRAPH ENTITIES BY TYPE")
    print("-" * 50)
    kg_entities = supabase.table("kg_entities")\
        .select("type", count="exact")\
        .execute()

    if kg_entities.data:
        type_counts = Counter([e['type'] for e in kg_entities.data])
        for etype, count in type_counts.most_common():
            print(f"  {etype}: {count}")
            if etype in ['ad_campaign', 'ad_platform']:
                print(f"    ✅ Ads in KG!")
    else:
        print("  ❌ No KG entities found")
    print()

    # 4. User profiles
    print("4. USER PROFILES")
    print("-" * 50)
    profiles = supabase.table("profiles")\
        .select("id, email, org_id")\
        .limit(5)\
        .execute()

    if profiles.data:
        for p in profiles.data:
            print(f"  {p['email']}: org_id = {p['org_id']}")
    print()

    # 5. Sample ads documents
    print("5. SAMPLE ADS DOCUMENTS")
    print("-" * 50)
    sample_ads = supabase.table("documents")\
        .select("id, org_id, source_type, title, embedding_status")\
        .in_("source_type", ["google_ads", "meta_ads"])\
        .limit(5)\
        .execute()

    if sample_ads.data:
        for doc in sample_ads.data:
            print(f"  {doc['source_type']}: {doc['title'][:50]}")
            print(f"    Org: {doc['org_id']}, Status: {doc['embedding_status']}")
    else:
        print("  ❌ No ads documents found!")
    print()

    # 6. Test get_document_stats function
    print("6. TEST get_document_stats FUNCTION")
    print("-" * 50)
    test_org_id = "00000000-0000-0000-0000-000000000001"
    try:
        stats = supabase.rpc("get_document_stats", {"org_uuid": test_org_id}).execute()
        if stats.data:
            print(f"  ✅ Function works! Results for test org:")
            for stat in stats.data:
                print(f"    {stat['source_type']}: {stat['total_documents']} total, {stat['completed_embeddings']} completed")
        else:
            print(f"  ⚠️  Function works but no documents for org {test_org_id}")
    except Exception as e:
        print(f"  ❌ Function error: {e}")
    print()

    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)

    # Check if there's a mismatch
    if ads_docs.data:
        ads_org = ads_docs.data[0]['org_id']
        if profiles.data:
            user_org = profiles.data[0]['org_id']
            if ads_org != user_org:
                print(f"⚠️  ORG MISMATCH DETECTED!")
                print(f"   Ads are in org: {ads_org}")
                print(f"   User profile in org: {user_org}")
                print()
                print("   FIX OPTIONS:")
                print(f"   1. Update user to test org:")
                print(f"      UPDATE profiles SET org_id = '{ads_org}' WHERE id = '{profiles.data[0]['id']}';")
                print()
                print(f"   2. Re-run backfill for user's real org:")
                print(f"      python scripts/backfill_campaign_embeddings.py --org-id {user_org}")
            else:
                print("✅ Org IDs match! Ads should be visible.")

    if not kg_entities.data or 'ad_campaign' not in [e['type'] for e in kg_entities.data]:
        print("⚠️  No ad_campaign entities in Knowledge Graph")
        print("   This means the trigger didn't fire for existing campaigns")
        print("   Run: backfill_existing_campaigns.sql to link them")

    print("=" * 80)


if __name__ == "__main__":
    diagnose()
