"""
ClickUp Service - OAuth and Data Fetching
Handles ClickUp OAuth 2.0 flow and data ingestion
"""

import os
import requests
from typing import Dict, List, Any, Optional
from supabase import Client


class ClickUpService:
    def __init__(self, supabase: Client):
        self.supabase = supabase
        self.client_id = os.getenv("CLICKUP_CLIENT_ID")
        self.client_secret = os.getenv("CLICKUP_CLIENT_SECRET")
        self.redirect_uri = os.getenv("CLICKUP_REDIRECT_URI", "http://localhost:8000/api/clickup/callback")
        self.base_url = "https://api.clickup.com/api/v2"

    def get_authorization_url(self, state: str) -> str:
        """Generate ClickUp OAuth authorization URL"""
        from urllib.parse import quote

        return (
            f"https://app.clickup.com/api?"
            f"client_id={self.client_id}&"
            f"redirect_uri={quote(self.redirect_uri, safe='')}&"
            f"state={state}"
        )

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token"""
        url = "https://api.clickup.com/api/v2/oauth/token"

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_authorized_user(self, access_token: str) -> Dict[str, Any]:
        """Get authorized user information"""
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.base_url}/user", headers=headers)
        response.raise_for_status()
        return response.json()

    def get_workspaces(self, access_token: str) -> List[Dict[str, Any]]:
        """Get user's ClickUp workspaces (teams)"""
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.base_url}/team", headers=headers)
        response.raise_for_status()
        return response.json().get("teams", [])

    def get_spaces(self, access_token: str, team_id: str) -> List[Dict[str, Any]]:
        """Get spaces in a workspace"""
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.base_url}/team/{team_id}/space", headers=headers)
        response.raise_for_status()
        return response.json().get("spaces", [])

    def get_lists(self, access_token: str, space_id: str) -> List[Dict[str, Any]]:
        """Get lists in a space"""
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.base_url}/space/{space_id}/list", headers=headers)
        response.raise_for_status()
        return response.json().get("lists", [])

    def get_tasks(self, access_token: str, list_id: str, archived: bool = False) -> List[Dict[str, Any]]:
        """Get tasks in a list"""
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"archived": str(archived).lower()}
        response = requests.get(
            f"{self.base_url}/list/{list_id}/task",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.json().get("tasks", [])

    def get_task_comments(self, access_token: str, task_id: str) -> List[Dict[str, Any]]:
        """Get comments for a task"""
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{self.base_url}/task/{task_id}/comment", headers=headers)
        response.raise_for_status()
        return response.json().get("comments", [])

    def create_webhook(
        self,
        access_token: str,
        team_id: str,
        endpoint: str,
        events: List[str]
    ) -> Dict[str, Any]:
        """Create a webhook for real-time updates"""
        headers = {"Authorization": f"Bearer {access_token}"}
        payload = {
            "endpoint": endpoint,
            "events": events
        }
        response = requests.post(
            f"{self.base_url}/team/{team_id}/webhook",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()

    def fetch_all_tasks(self, connection_id: str, org_id: str) -> List[Dict[str, Any]]:
        """Fetch all tasks from connected ClickUp workspace"""
        # Get connection details
        conn_result = self.supabase.table("connections")\
            .select("*")\
            .eq("id", connection_id)\
            .eq("org_id", org_id)\
            .single()\
            .execute()

        if not conn_result.data:
            raise ValueError(f"Connection {connection_id} not found")

        connection = conn_result.data
        print(f"[ClickUp] Connection data: {connection}")

        access_token = connection.get("access_token")
        if not access_token:
            raise ValueError("No access_token found in connection")

        workspace_id = connection.get("workspace_id")
        if not workspace_id:
            raise ValueError(f"No workspace_id found in connection. Connection data: {connection}")

        all_tasks = []

        # Get all spaces in the workspace
        print(f"[ClickUp] Fetching spaces for workspace {workspace_id}")
        try:
            spaces = self.get_spaces(access_token, workspace_id)
            print(f"[ClickUp] Found {len(spaces)} spaces")
        except Exception as e:
            print(f"[ClickUp] ERROR fetching spaces: {type(e).__name__}: {e}")
            raise

        for space in spaces:
            space_id = space.get("id")
            space_name = space.get("name")
            print(f"[ClickUp] Processing space: {space_name} ({space_id})")

            # Get all lists in the space
            try:
                lists = self.get_lists(access_token, space_id)
                print(f"[ClickUp] Found {len(lists)} lists in space {space_name}")
            except Exception as e:
                print(f"[ClickUp] ERROR fetching lists for space {space_id}: {type(e).__name__}: {e}")
                continue

            for list_item in lists:
                list_id = list_item.get("id")
                list_name = list_item.get("name")
                print(f"[ClickUp] Processing list: {list_name} ({list_id})")

                # Get all tasks in the list
                try:
                    tasks = self.get_tasks(access_token, list_id)
                    print(f"[ClickUp] Found {len(tasks)} tasks in list {list_name}")
                except Exception as e:
                    print(f"[ClickUp] ERROR fetching tasks for list {list_id}: {type(e).__name__}: {e}")
                    continue

                for task in tasks:
                    task_id = task.get("id")
                    task_name = task.get("name")
                    task_data = {
                        "task": task,
                        "space_name": space_name,
                        "list_name": list_name,
                        "comments": []
                    }

                    # Fetch comments for the task
                    try:
                        comments = self.get_task_comments(access_token, task_id)
                        task_data["comments"] = comments
                    except Exception as e:
                        print(f"[ClickUp] Failed to fetch comments for task {task_id}: {e}")

                    all_tasks.append(task_data)
                    print(f"[ClickUp] Added task: {task_name}")

        return all_tasks

    def normalize_task_to_document(
        self,
        task_data: Dict[str, Any],
        org_id: str,
        connection_id: str
    ) -> Dict[str, Any]:
        """Normalize ClickUp task to document format"""
        task = task_data["task"]
        space_name = task_data["space_name"]
        list_name = task_data["list_name"]
        comments = task_data["comments"]

        # Build content
        status = task.get('status', {})
        status_text = status.get('status', 'Unknown') if status else 'Unknown'

        content_parts = [
            f"Task: {task.get('name', 'Untitled')}",
            f"Space: {space_name}",
            f"List: {list_name}",
            f"Status: {status_text}",
        ]

        if task.get("description"):
            content_parts.append(f"Description: {task['description']}")

        if task.get("assignees"):
            assignees = ", ".join([a.get("username", "Unknown") for a in task["assignees"]])
            content_parts.append(f"Assignees: {assignees}")

        if task.get("priority"):
            priority_obj = task.get("priority", {})
            priority_val = priority_obj.get("priority") if priority_obj else None
            if priority_val:
                priority_map = {1: "Urgent", 2: "High", 3: "Normal", 4: "Low"}
                priority = priority_map.get(priority_val, "None")
                content_parts.append(f"Priority: {priority}")

        if task.get("due_date"):
            content_parts.append(f"Due Date: {task['due_date']}")

        # Add comments
        if comments:
            content_parts.append("\nComments:")
            for comment in comments:
                user = comment.get("user", {}).get("username", "Unknown")
                text = comment.get("comment_text", "")
                content_parts.append(f"- {user}: {text}")

        content = "\n".join(content_parts)

        # Build metadata
        status_obj = task.get("status", {})
        priority_obj = task.get("priority", {})
        creator_obj = task.get("creator", {})

        metadata = {
            "task_id": task.get("id"),
            "task_url": task.get("url"),
            "space_name": space_name,
            "list_name": list_name,
            "status": status_obj.get("status") if status_obj else None,
            "priority": priority_obj.get("priority") if priority_obj else None,
            "assignees": [a.get("username") for a in task.get("assignees", [])],
            "due_date": task.get("due_date"),
            "tags": [tag.get("name") for tag in task.get("tags", [])],
        }

        # Build document
        document = {
            "org_id": org_id,
            "connection_id": connection_id,
            "source_type": "clickup",
            "source_id": task.get("id"),
            "title": task.get("name", "Untitled"),
            "content": content,
            "url": task.get("url"),
            "metadata": metadata,
            "author": creator_obj.get("username", "Unknown") if creator_obj else "Unknown",
            "author_id": str(creator_obj.get("id", "")) if creator_obj else "",
        }

        return document


# Singleton pattern
_clickup_service = None

def get_clickup_service(supabase: Client) -> ClickUpService:
    """Get or create ClickUpService instance"""
    global _clickup_service
    if _clickup_service is None:
        _clickup_service = ClickUpService(supabase)
    return _clickup_service
