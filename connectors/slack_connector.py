# -*- coding: utf-8 -*-
"""
Slack Connector
Handles Slack OAuth and data ingestion
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from slack_sdk import WebClient
from slack_sdk.oauth import AuthorizeUrlGenerator
from slack_sdk.errors import SlackApiError


class SlackConnector:
    def __init__(self):
        """Initialize Slack connector"""
        self.client_id = os.getenv("SLACK_CLIENT_ID")
        self.client_secret = os.getenv("SLACK_CLIENT_SECRET")
        self.redirect_uri = os.getenv("SLACK_REDIRECT_URI", "http://localhost:8000/api/connect/slack/callback")

        if not self.client_id or not self.client_secret:
            raise ValueError("SLACK_CLIENT_ID and SLACK_CLIENT_SECRET must be set")

        self.scopes = [
            "channels:history",
            "channels:read",
            "users:read",
            "team:read"
        ]

    def get_authorization_url(self, state: str = None) -> str:
        """
        Generate Slack OAuth authorization URL

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL
        """
        url_generator = AuthorizeUrlGenerator(
            client_id=self.client_id,
            scopes=self.scopes,
            redirect_uri=self.redirect_uri
        )

        return url_generator.generate(state=state)

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """
        Exchange OAuth code for access token

        Args:
            code: OAuth authorization code

        Returns:
            Dictionary containing access_token, team info, etc.
        """
        from slack_sdk.oauth import OAuthStateUtils

        client = WebClient()

        response = client.oauth_v2_access(
            client_id=self.client_id,
            client_secret=self.client_secret,
            code=code,
            redirect_uri=self.redirect_uri
        )

        return {
            "access_token": response["access_token"],
            "token_type": response["token_type"],
            "scope": response["scope"],
            "team_id": response["team"]["id"],
            "team_name": response["team"]["name"],
            "authed_user": response.get("authed_user", {})
        }

    def get_workspace_info(self, access_token: str) -> Dict[str, Any]:
        """
        Get Slack workspace information

        Args:
            access_token: Slack access token

        Returns:
            Workspace info
        """
        client = WebClient(token=access_token)

        try:
            team_info = client.team_info()
            return {
                "id": team_info["team"]["id"],
                "name": team_info["team"]["name"],
                "domain": team_info["team"]["domain"]
            }
        except SlackApiError as e:
            raise Exception(f"Failed to get workspace info: {e.response['error']}")

    def list_channels(self, access_token: str, types: str = "public_channel", auto_join: bool = True) -> List[Dict[str, Any]]:
        """
        List channels in workspace and optionally auto-join public channels

        Args:
            access_token: Slack access token
            types: Channel types (public_channel, private_channel, mpim, im)
            auto_join: Automatically join public channels (default: True)

        Returns:
            List of channel dictionaries
        """
        client = WebClient(token=access_token)

        channels = []
        cursor = None

        try:
            while True:
                response = client.conversations_list(
                    types=types,
                    limit=200,
                    cursor=cursor,
                    exclude_archived=True  # Skip archived channels
                )

                channels.extend(response["channels"])

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            # Process channels and auto-join public ones if needed
            result_channels = []
            for ch in channels:
                channel_info = {
                    "id": ch["id"],
                    "name": ch["name"],
                    "is_private": ch.get("is_private", False),
                    "num_members": ch.get("num_members", 0)
                }

                # If already a member, include it
                if ch.get("is_member", False):
                    result_channels.append(channel_info)
                # If it's a public channel and auto_join is enabled, try to join
                elif auto_join and not ch.get("is_private", False):
                    try:
                        client.conversations_join(channel=ch["id"])
                        print(f"✓ Auto-joined public channel: #{ch['name']}")
                        result_channels.append(channel_info)
                    except SlackApiError as join_error:
                        print(f"✗ Could not join #{ch['name']}: {join_error.response['error']}")
                # Skip private channels bot is not a member of
                elif ch.get("is_private", False):
                    print(f"⊘ Skipping private channel #{ch['name']} (bot not invited)")

            return result_channels

        except SlackApiError as e:
            raise Exception(f"Failed to list channels: {e.response['error']}")

    def fetch_messages(
        self,
        access_token: str,
        channel_id: str,
        days_back: int = 30,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Fetch messages from a Slack channel

        Args:
            access_token: Slack access token
            channel_id: Channel ID to fetch from
            days_back: How many days back to fetch
            limit: Maximum messages to fetch

        Returns:
            List of message dictionaries
        """
        client = WebClient(token=access_token)

        # Calculate oldest timestamp
        oldest = (datetime.now() - timedelta(days=days_back)).timestamp()

        messages = []
        cursor = None

        try:
            while len(messages) < limit:
                response = client.conversations_history(
                    channel=channel_id,
                    oldest=str(oldest),
                    limit=min(200, limit - len(messages)),
                    cursor=cursor
                )

                batch = response["messages"]
                messages.extend(batch)

                cursor = response.get("response_metadata", {}).get("next_cursor")
                if not cursor or not batch:
                    break

            # Filter out bot messages and system messages
            filtered_messages = []
            for msg in messages:
                if msg.get("subtype"):  # Skip system messages
                    continue
                if msg.get("bot_id"):  # Skip bot messages
                    continue

                # Extract full message content including blocks and attachments
                full_text = msg.get("text", "")

                # Replace user mentions with names (if we can resolve them)
                # Slack format: <@U1234567> should become @username
                # We'll keep the raw format for now since resolving would require extra API calls
                # But we'll clean up the formatting to make it more readable
                if full_text:
                    # Remove angle brackets from mentions
                    import re
                    full_text = re.sub(r'<@(U[A-Z0-9]+)>', r'@\1', full_text)
                    full_text = re.sub(r'<#(C[A-Z0-9]+)\|([^>]+)>', r'#\2', full_text)  # Channel mentions
                    full_text = re.sub(r'<(https?://[^>]+)>', r'\1', full_text)  # URLs

                # Extract text from blocks (rich formatting)
                if msg.get("blocks"):
                    block_texts = []
                    for block in msg["blocks"]:
                        if block.get("type") == "rich_text":
                            # Extract from rich_text elements
                            for element in block.get("elements", []):
                                if element.get("type") == "rich_text_section":
                                    for item in element.get("elements", []):
                                        if item.get("type") == "text":
                                            block_texts.append(item.get("text", ""))
                        elif block.get("type") == "section" and block.get("text"):
                            block_texts.append(block["text"].get("text", ""))

                    if block_texts:
                        full_text = full_text + "\n" + "\n".join(block_texts) if full_text else "\n".join(block_texts)

                # Extract attachment text
                if msg.get("attachments"):
                    attachment_texts = []
                    for att in msg["attachments"]:
                        if att.get("text"):
                            attachment_texts.append(att["text"])
                        if att.get("pretext"):
                            attachment_texts.append(att["pretext"])
                        if att.get("fallback"):
                            attachment_texts.append(att["fallback"])

                    if attachment_texts:
                        full_text = full_text + "\n" + "\n".join(attachment_texts) if full_text else "\n".join(attachment_texts)

                # Extract file information
                if msg.get("files"):
                    file_info = []
                    for file in msg["files"]:
                        file_name = file.get("name", "")
                        file_title = file.get("title", "")
                        if file_name or file_title:
                            file_info.append(f"[File: {file_title or file_name}]")

                    if file_info:
                        full_text = full_text + "\n" + "\n".join(file_info) if full_text else "\n".join(file_info)

                filtered_messages.append({
                    "ts": msg["ts"],
                    "user": msg.get("user"),
                    "text": full_text.strip(),
                    "thread_ts": msg.get("thread_ts"),
                    "reply_count": msg.get("reply_count", 0)
                })

            return filtered_messages

        except SlackApiError as e:
            error_code = e.response['error']
            if error_code == 'not_in_channel':
                raise Exception(f"Bot is not in channel {channel_id}. Please invite the bot to the channel using '/invite @YourApp' in Slack, or add 'channels:history' scope to your Slack app.")
            else:
                raise Exception(f"Failed to fetch messages: {error_code}")

    def fetch_thread_replies(
        self,
        access_token: str,
        channel_id: str,
        thread_ts: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch replies in a thread

        Args:
            access_token: Slack access token
            channel_id: Channel ID
            thread_ts: Thread timestamp

        Returns:
            List of reply messages
        """
        client = WebClient(token=access_token)

        try:
            response = client.conversations_replies(
                channel=channel_id,
                ts=thread_ts
            )

            # Skip the parent message (first in list)
            replies = response["messages"][1:] if len(response["messages"]) > 1 else []

            # Extract text from replies
            reply_texts = []
            for reply in replies:
                if reply.get("bot_id"):  # Skip bot messages
                    continue

                text = reply.get("text", "")

                # Extract from blocks if present
                if reply.get("blocks"):
                    block_texts = []
                    for block in reply["blocks"]:
                        if block.get("type") == "rich_text":
                            for element in block.get("elements", []):
                                if element.get("type") == "rich_text_section":
                                    for item in element.get("elements", []):
                                        if item.get("type") == "text":
                                            block_texts.append(item.get("text", ""))
                        elif block.get("type") == "section" and block.get("text"):
                            block_texts.append(block["text"].get("text", ""))

                    if block_texts:
                        text = text + "\n" + "\n".join(block_texts) if text else "\n".join(block_texts)

                if text.strip():
                    reply_texts.append(text.strip())

            return reply_texts

        except SlackApiError as e:
            print(f"Failed to fetch thread replies: {e.response['error']}")
            return []

    def get_user_info(self, access_token: str, user_id: str) -> Dict[str, Any]:
        """
        Get user information

        Args:
            access_token: Slack access token
            user_id: User ID

        Returns:
            User info dictionary
        """
        client = WebClient(token=access_token)

        try:
            response = client.users_info(user=user_id)
            user = response["user"]

            return {
                "id": user["id"],
                "name": user.get("name", ""),
                "real_name": user.get("real_name", ""),
                "display_name": user.get("profile", {}).get("display_name", "")
            }
        except SlackApiError:
            return {"id": user_id, "name": "Unknown User", "real_name": "", "display_name": ""}


# Singleton instance
_slack_connector = None


def get_slack_connector() -> SlackConnector:
    """Get or create Slack connector singleton"""
    global _slack_connector
    if _slack_connector is None:
        _slack_connector = SlackConnector()
    return _slack_connector
