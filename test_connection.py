#!/usr/bin/env python3
"""
Quick test to verify Supabase connection and table setup
"""

from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print("🔍 Testing Supabase Connection...")
print(f"URL: {SUPABASE_URL}")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Test 1: Organizations table
try:
    result = supabase.table("organizations").select("*", count="exact").execute()
    print(f"✅ Organizations table: {result.count} records")
except Exception as e:
    print(f"❌ Organizations table error: {e}")

# Test 2: Profiles table
try:
    result = supabase.table("profiles").select("*", count="exact").execute()
    print(f"✅ Profiles table: {result.count} records")
except Exception as e:
    print(f"❌ Profiles table error: {e}")

# Test 3: Documents table
try:
    result = supabase.table("documents").select("*", count="exact").execute()
    print(f"✅ Documents table: {result.count} records")
except Exception as e:
    print(f"❌ Documents table error: {e}")

# Test 4: Embeddings table
try:
    result = supabase.table("embeddings").select("*", count="exact").execute()
    print(f"✅ Embeddings table: {result.count} records")
except Exception as e:
    print(f"❌ Embeddings table error: {e}")

# Test 5: Feedback table
try:
    result = supabase.table("feedback").select("*", count="exact").execute()
    print(f"✅ Feedback table: {result.count} records")
except Exception as e:
    print(f"❌ Feedback table error: {e}")

print("\n✅ Connection test complete!")
