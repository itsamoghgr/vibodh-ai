"""
Test script for Marketing Agent
Run this after starting the backend to test the marketing agent functionality
"""

import requests
import json
from uuid import uuid4

# Configuration
BASE_URL = "http://localhost:8000"
ORG_ID = str(uuid4())  # Replace with actual org ID
USER_ID = str(uuid4())  # Replace with actual user ID


def test_marketing_agent_via_orchestrator():
    """Test marketing agent through the orchestrator"""
    print("\n=== Testing Marketing Agent via Orchestrator ===\n")

    # Test query that should trigger marketing agent
    query = "send a slack message to the private-ch channel that We will be starting with the marketing campaign soon, lets have a meeting this friday."

    payload = {
        "query": query,
        "org_id": ORG_ID,
        "user_id": USER_ID
    }

    print(f"Query: {query}")
    print(f"Sending to orchestrator...")

    response = requests.post(
        f"{BASE_URL}/api/v1/orchestrate/query",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Success!")
        print(f"Intent: {result.get('intent')}")
        print(f"Modules Used: {result.get('modules_used')}")
        print(f"\nFinal Answer:\n{result.get('final_answer')}")

        if result.get('module_results', {}).get('agent'):
            agent_result = result['module_results']['agent']
            print(f"\n=== Agent Result ===")
            print(f"Plan Created: {agent_result.get('plan_created')}")
            print(f"Plan ID: {agent_result.get('plan_id')}")
            print(f"Risk Level: {agent_result.get('risk_level')}")
            print(f"Requires Approval: {agent_result.get('requires_approval')}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)


def test_direct_marketing_execution():
    """Test direct marketing agent execution"""
    print("\n=== Testing Direct Marketing Execution ===\n")

    payload = {
        "agent_type": "marketing_agent",
        "goal": "Create and launch a product announcement campaign for our new AI feature",
        "context": {
            "product": "AI Knowledge Graph",
            "launch_date": "2025-02-01",
            "target_channels": ["slack", "email"]
        },
        "auto_approve": False,
        "dry_run": False
    }

    print(f"Goal: {payload['goal']}")
    print(f"Executing marketing agent...")

    response = requests.post(
        f"{BASE_URL}/api/v1/agents/execute?org_id={ORG_ID}&user_id={USER_ID}",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Success!")
        print(f"Plan ID: {result.get('plan_id')}")
        print(f"Status: {result.get('status')}")
        print(f"Risk Level: {result.get('risk_level')}")
        print(f"Total Steps: {result.get('total_steps')}")
        print(f"Requires Approval: {result.get('requires_approval')}")
        print(f"Message: {result.get('message')}")
        print(f"Next Action: {result.get('next_action')}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)


def test_content_generation():
    """Test marketing content generation"""
    print("\n=== Testing Content Generation ===\n")

    payload = {
        "content_type": "social_post",
        "topic": "AI-powered knowledge management launch",
        "tone": "professional",
        "length": "short",
        "target_audience": "Tech professionals and business leaders",
        "key_points": [
            "Revolutionary AI technology",
            "Seamless integration",
            "Boost productivity"
        ],
        "include_cta": True,
        "include_hashtags": True
    }

    print(f"Content Type: {payload['content_type']}")
    print(f"Topic: {payload['topic']}")
    print(f"Generating content...")

    response = requests.post(
        f"{BASE_URL}/api/v1/marketing/content/generate?org_id={ORG_ID}",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        content = result.get('content', {})
        print(f"\n✅ Generated Content:")
        print(f"Title: {content.get('title')}")
        print(f"Body: {content.get('body')}")
        print(f"CTA: {content.get('call_to_action')}")
        print(f"Hashtags: {', '.join(content.get('hashtags', []))}")
        print(f"Generation Time: {result.get('generation_time_ms')}ms")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)


def test_campaign_creation():
    """Test campaign creation"""
    print("\n=== Testing Campaign Creation ===\n")

    payload = {
        "title": "Q1 Product Launch Campaign",
        "description": "Launch our new AI-powered features to the market",
        "campaign_type": "product_launch",
        "goals": [
            "Generate 500 qualified leads",
            "Achieve 10% engagement rate",
            "Build brand awareness"
        ],
        "target_audience": "Tech professionals and enterprise decision makers",
        "channels": ["slack", "email", "social_media"],
        "timeline_days": 30,
        "key_messages": [
            "Revolutionary AI technology",
            "Enterprise-ready solution",
            "Seamless integration"
        ],
        "success_metrics": [
            "Lead generation count",
            "Engagement rate",
            "Click-through rate"
        ],
        "estimated_reach": "5000-10000"
    }

    print(f"Campaign: {payload['title']}")
    print(f"Creating campaign...")

    response = requests.post(
        f"{BASE_URL}/api/v1/marketing/campaigns?org_id={ORG_ID}&auto_execute=True",
        json=payload
    )

    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Campaign Created!")
        print(f"Campaign ID: {result.get('id')}")
        print(f"Status: {result.get('status')}")
        print(f"Type: {result.get('campaign_type')}")
        print(f"Timeline: {result.get('timeline_days')} days")
        print(f"Channels: {', '.join(result.get('channels', []))}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)


def check_agent_registry():
    """Check if marketing agent is registered"""
    print("\n=== Checking Agent Registry ===\n")

    response = requests.get(
        f"{BASE_URL}/api/v1/agents/registry?org_id={ORG_ID}"
    )

    if response.status_code == 200:
        agents = response.json()
        print(f"Registered Agents: {len(agents)}")
        for agent in agents:
            print(f"\n- {agent.get('agent_name')} ({agent.get('agent_type')})")
            print(f"  Status: {agent.get('status')}")
            print(f"  Capabilities: {', '.join(agent.get('capabilities', []))}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    print("""
    ================================
    Marketing Agent Test Suite
    ================================

    Make sure the backend is running:
    cd vibodh-ai && uvicorn app.main:app --reload --port 8000

    Tests to run:
    1. Check agent registry
    2. Test via orchestrator (your original query)
    3. Test direct execution
    4. Test content generation
    5. Test campaign creation
    """)

    # Run tests
    try:
        # Check registry first
        check_agent_registry()

        # Test orchestrator integration
        test_marketing_agent_via_orchestrator()

        # Test direct execution
        test_direct_marketing_execution()

        # Test content generation
        test_content_generation()

        # Test campaign creation
        test_campaign_creation()

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to backend. Make sure it's running on port 8000")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")