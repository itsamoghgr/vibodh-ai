"""
Slack Service
Low-level Slack API integration service
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from app.core.logging import logger
import requests


class SlackService:
    """
    Service for interacting with Slack API.

    Handles:
    - OAuth authentication
    - Channel management
    - Message posting
    - History retrieval
    """

    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize Slack service.

        Args:
            client_id: Slack app client ID
            client_secret: Slack app client secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://slack.com/api"

        logger.info("[SLACK_SERVICE] Service initialized")

    def exchange_code(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """
        Exchange OAuth code for access token.

        Args:
            code: OAuth authorization code
            redirect_uri: OAuth redirect URI

        Returns:
            Token response with access token
        """
        try:
            logger.info("[SLACK_SERVICE] Exchanging OAuth code for token")

            response = requests.post(
                f"{self.base_url}/oauth.v2.access",
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    logger.info("[SLACK_SERVICE] Token exchange successful")
                    return data
                else:
                    error = data.get("error", "Unknown error")
                    logger.error(f"[SLACK_SERVICE] Token exchange failed: {error}")
                    raise Exception(f"Slack OAuth failed: {error}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to exchange code: {e}")
            raise

    def get_channels(self, access_token: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of channels in workspace.

        Args:
            access_token: Slack access token
            limit: Maximum number of channels

        Returns:
            List of channel objects
        """
        try:
            logger.info("[SLACK_SERVICE] Fetching channels")

            response = requests.get(
                f"{self.base_url}/conversations.list",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "types": "public_channel,private_channel",
                    "limit": limit
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    channels = data.get("channels", [])
                    logger.info(f"[SLACK_SERVICE] Found {len(channels)} channels")
                    return channels
                else:
                    error = data.get("error", "Unknown error")
                    logger.error(f"[SLACK_SERVICE] Failed to get channels: {error}")
                    return []
            else:
                logger.error(f"[SLACK_SERVICE] HTTP {response.status_code}: {response.text}")
                return []

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to get channels: {e}")
            return []

    def get_channel_history(
        self,
        access_token: str,
        channel_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get message history for a channel.

        Args:
            access_token: Slack access token
            channel_id: Channel ID
            limit: Maximum number of messages

        Returns:
            List of message objects
        """
        try:
            logger.info(f"[SLACK_SERVICE] Fetching history for channel {channel_id}")

            response = requests.get(
                f"{self.base_url}/conversations.history",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "channel": channel_id,
                    "limit": limit
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    messages = data.get("messages", [])
                    logger.info(f"[SLACK_SERVICE] Found {len(messages)} messages")
                    return messages
                else:
                    error = data.get("error", "Unknown error")
                    logger.error(f"[SLACK_SERVICE] Failed to get history: {error}")
                    return []
            else:
                logger.error(f"[SLACK_SERVICE] HTTP {response.status_code}: {response.text}")
                return []

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to get history: {e}")
            return []

    def post_message(
        self,
        access_token: str,
        channel: str,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Post a message to a Slack channel.

        Args:
            access_token: Slack access token
            channel: Channel ID or name
            text: Message text
            blocks: Optional block elements for rich formatting

        Returns:
            Response from Slack API
        """
        try:
            logger.info(f"[SLACK_SERVICE] Posting message to {channel}")

            payload = {
                "channel": channel,
                "text": text
            }

            if blocks:
                payload["blocks"] = blocks

            response = requests.post(
                f"{self.base_url}/chat.postMessage",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    logger.info(f"[SLACK_SERVICE] Message posted successfully")
                    return data
                else:
                    error = data.get("error", "Unknown error")
                    logger.error(f"[SLACK_SERVICE] Failed to post message: {error}")
                    raise Exception(f"Slack post failed: {error}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to post message: {e}")
            raise

    def schedule_message(
        self,
        access_token: str,
        channel: str,
        text: str,
        post_at: datetime,
        blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Schedule a message for later posting.

        Args:
            access_token: Slack access token
            channel: Channel ID or name
            text: Message text
            post_at: When to post the message
            blocks: Optional block elements

        Returns:
            Response from Slack API
        """
        try:
            logger.info(f"[SLACK_SERVICE] Scheduling message for {post_at}")

            # Convert datetime to Unix timestamp
            post_at_timestamp = int(post_at.timestamp())

            payload = {
                "channel": channel,
                "text": text,
                "post_at": post_at_timestamp
            }

            if blocks:
                payload["blocks"] = blocks

            response = requests.post(
                f"{self.base_url}/chat.scheduleMessage",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    logger.info(f"[SLACK_SERVICE] Message scheduled successfully")
                    return data
                else:
                    error = data.get("error", "Unknown error")
                    logger.error(f"[SLACK_SERVICE] Failed to schedule message: {error}")
                    raise Exception(f"Slack schedule failed: {error}")
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to schedule message: {e}")
            raise

    def get_user_info(self, access_token: str, user_id: str) -> Dict[str, Any]:
        """
        Get information about a Slack user.

        Args:
            access_token: Slack access token
            user_id: User ID

        Returns:
            User information
        """
        try:
            logger.info(f"[SLACK_SERVICE] Getting info for user {user_id}")

            response = requests.get(
                f"{self.base_url}/users.info",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"user": user_id}
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    return data.get("user", {})
                else:
                    error = data.get("error", "Unknown error")
                    logger.error(f"[SLACK_SERVICE] Failed to get user info: {error}")
                    return {}
            else:
                logger.error(f"[SLACK_SERVICE] HTTP {response.status_code}: {response.text}")
                return {}

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to get user info: {e}")
            return {}

    def test_auth(self, access_token: str) -> bool:
        """
        Test if access token is valid.

        Args:
            access_token: Slack access token

        Returns:
            True if token is valid
        """
        try:
            response = requests.get(
                f"{self.base_url}/auth.test",
                headers={"Authorization": f"Bearer {access_token}"}
            )

            if response.status_code == 200:
                data = response.json()
                is_valid = data.get("ok", False)

                if is_valid:
                    logger.info(f"[SLACK_SERVICE] Auth test successful for {data.get('team', 'unknown')}")
                else:
                    logger.warning(f"[SLACK_SERVICE] Auth test failed: {data.get('error', 'Unknown')}")

                return is_valid
            else:
                logger.error(f"[SLACK_SERVICE] Auth test HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Auth test failed: {e}")
            return False

    def list_channels(
        self,
        access_token: str,
        types: str = "public_channel",
        auto_join: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List channels in workspace.

        Args:
            access_token: Slack access token
            types: Channel types to fetch
            auto_join: Whether to auto-join channels

        Returns:
            List of channel objects
        """
        try:
            response = requests.get(
                f"{self.base_url}/conversations.list",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"types": types, "limit": 1000}
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    return data.get("channels", [])
                else:
                    error = data.get("error", "Unknown error")
                    raise Exception(error)
            else:
                raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to list channels: {e}")
            raise

    def fetch_messages(
        self,
        access_token: str,
        channel_id: str,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Fetch messages from a channel.

        Args:
            access_token: Slack access token
            channel_id: Channel ID
            days_back: How many days back to fetch

        Returns:
            List of messages
        """
        try:
            oldest_timestamp = (datetime.utcnow() - timedelta(days=days_back)).timestamp()

            response = requests.get(
                f"{self.base_url}/conversations.history",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "channel": channel_id,
                    "oldest": oldest_timestamp,
                    "limit": 1000
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    return data.get("messages", [])
                else:
                    error = data.get("error", "Unknown error")
                    raise Exception(error)
            else:
                raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to fetch messages: {e}")
            raise

    def fetch_thread_replies(
        self,
        access_token: str,
        channel_id: str,
        thread_ts: str
    ) -> List[str]:
        """
        Fetch thread replies for a message.

        Args:
            access_token: Slack access token
            channel_id: Channel ID
            thread_ts: Thread timestamp

        Returns:
            List of reply texts
        """
        try:
            response = requests.get(
                f"{self.base_url}/conversations.replies",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "channel": channel_id,
                    "ts": thread_ts
                }
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    messages = data.get("messages", [])[1:]  # Skip the parent message
                    return [msg.get("text", "") for msg in messages]
                else:
                    return []
            else:
                return []

        except Exception as e:
            logger.error(f"[SLACK_SERVICE] Failed to fetch thread replies: {e}")
            return []