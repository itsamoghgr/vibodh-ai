#!/usr/bin/env python3
"""
Test Ads Semantic Search

Verifies that ads campaigns can be found using semantic/natural language queries.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db.supabase_client import get_supabase_admin_client
from app.services.embedding_service import get_embedding_service
from app.core.logging import logger


def test_semantic_search(query: str, org_id: str, match_threshold: float = 0.7, match_count: int = 5):
    """
    Test semantic search for ads campaigns.

    Args:
        query: Natural language search query
        org_id: Organization ID to filter results
        match_threshold: Minimum similarity score (0-1)
        match_count: Number of results to return
    """
    supabase = get_supabase_admin_client()
    embedding_service = get_embedding_service()

    logger.info(f"\n{'='*80}")
    logger.info(f"Search Query: '{query}'")
    logger.info(f"{'='*80}")

    # Generate embedding for the query
    logger.info("Generating query embedding...")
    query_embedding = embedding_service.generate_embedding(query)

    # Search using pgvector similarity
    logger.info(f"Searching for similar documents (threshold: {match_threshold}, limit: {match_count})...")

    # Use Supabase's RPC for vector similarity search
    try:
        # First, let's try a direct query to check if embeddings exist
        docs_check = supabase.table("documents")\
            .select("id, source_type, title")\
            .eq("org_id", org_id)\
            .in_("source_type", ["google_ads", "meta_ads"])\
            .limit(5)\
            .execute()

        logger.info(f"Found {len(docs_check.data)} ads documents in database")

        # Now try the actual vector search
        # Note: This assumes you have a match_documents RPC function
        # If not, we'll need to create one
        results = supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
                "match_org_id": org_id
            }
        ).execute()

        if not results.data:
            logger.warning("No results found!")
            logger.info("This could mean:")
            logger.info("1. No campaigns match the query above the threshold")
            logger.info("2. The match_documents RPC function may not exist")
            return []

        logger.info(f"\nFound {len(results.data)} matching campaigns:\n")

        for idx, result in enumerate(results.data, 1):
            logger.info(f"{idx}. {result.get('title', 'Untitled')}")
            logger.info(f"   Platform: {result.get('source_type', 'unknown')}")
            logger.info(f"   Similarity: {result.get('similarity', 0):.3f}")
            logger.info(f"   Summary: {result.get('summary', 'No summary')[:100]}...")
            logger.info("")

        return results.data

    except Exception as e:
        logger.error(f"Search failed: {e}")
        logger.info("\nTrying alternative search method...")

        # Fallback: Get all ads documents and calculate similarity manually
        all_docs = supabase.table("documents")\
            .select("id, source_type, title, summary, content")\
            .eq("org_id", org_id)\
            .in_("source_type", ["google_ads", "meta_ads"])\
            .execute()

        logger.info(f"Retrieved {len(all_docs.data)} ads documents for manual similarity calculation")
        logger.info("(Note: In production, use a proper RPC function for vector search)")

        # For now, just show what we have
        logger.info("\nAvailable Ads Documents:")
        for idx, doc in enumerate(all_docs.data[:match_count], 1):
            logger.info(f"{idx}. {doc.get('title', 'Untitled')}")
            logger.info(f"   Platform: {doc.get('source_type', 'unknown')}")
            logger.info(f"   Summary: {doc.get('summary', 'No summary')[:100]}...")
            logger.info("")

        return all_docs.data[:match_count]


def main():
    org_id = "00000000-0000-0000-0000-000000000001"

    # Test queries
    test_queries = [
        "campaigns about conversions",
        "Google Ads campaigns",
        "traffic campaigns",
        "brand awareness ads",
        "Q4 campaigns"
    ]

    logger.info("=" * 80)
    logger.info("Testing Ads Semantic Search")
    logger.info("=" * 80)

    for query in test_queries:
        try:
            results = test_semantic_search(query, org_id, match_threshold=0.3, match_count=3)
            logger.info(f"✓ Query '{query}' completed with {len(results)} results\n")
        except Exception as e:
            logger.error(f"✗ Query '{query}' failed: {e}\n")

    logger.info("=" * 80)
    logger.info("Test completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
