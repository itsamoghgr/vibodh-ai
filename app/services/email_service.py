"""
Email Service with SMTP Support

Provides email sending capabilities for:
- Communication agent backup messages
- System notifications and alerts
- Campaign summaries and reports
- User invitations and onboarding

Features:
- SMTP with TLS/SSL support
- HTML and plain text templates
- Attachment support
- Rate limiting and retry logic
- Delivery tracking
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
from pathlib import Path

from app.core.config import settings
from app.core.logging import logger, log_error


class EmailConfig:
    """Email configuration settings"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_username: str,
        smtp_password: str,
        from_email: str,
        from_name: str = "Vibodh AI",
        use_tls: bool = True,
        use_ssl: bool = False
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.from_name = from_name
        self.use_tls = use_tls
        self.use_ssl = use_ssl


class EmailMessage:
    """Email message container"""

    def __init__(
        self,
        to: Union[str, List[str]],
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        reply_to: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ):
        self.to = [to] if isinstance(to, str) else to
        self.subject = subject
        self.body_text = body_text
        self.body_html = body_html
        self.cc = cc or []
        self.bcc = bcc or []
        self.reply_to = reply_to
        self.attachments = attachments or []


class EmailService:
    """
    Production-ready email service with SMTP support.

    Features:
    - Configurable SMTP with TLS/SSL
    - HTML and plain text support
    - Attachment handling
    - Retry logic with exponential backoff
    - Delivery tracking
    """

    def __init__(self, config: EmailConfig, supabase=None):
        """
        Initialize email service.

        Args:
            config: Email configuration
            supabase: Optional Supabase client for delivery tracking
        """
        self.config = config
        self.supabase = supabase
        self._validate_config()
        logger.info(f"[EMAIL_SERVICE] Initialized with host {config.smtp_host}:{config.smtp_port}")

    def _validate_config(self):
        """Validate email configuration"""
        required_fields = [
            'smtp_host', 'smtp_port', 'smtp_username',
            'smtp_password', 'from_email'
        ]

        for field in required_fields:
            if not getattr(self.config, field):
                raise ValueError(f"Email config missing required field: {field}")

    def send_email(
        self,
        message: EmailMessage,
        org_id: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Send an email with retry logic.

        Args:
            message: EmailMessage to send
            org_id: Organization ID for tracking
            max_retries: Maximum number of retry attempts

        Returns:
            Dict with success status and delivery details
        """
        attempt = 0
        last_error = None

        while attempt < max_retries:
            try:
                attempt += 1
                logger.info(
                    f"[EMAIL_SERVICE] Sending email (attempt {attempt}/{max_retries})",
                    extra={
                        "to": message.to,
                        "subject": message.subject,
                        "org_id": org_id
                    }
                )

                result = self._send_via_smtp(message)

                # Track successful delivery
                if self.supabase and org_id:
                    self._track_delivery(
                        org_id=org_id,
                        message=message,
                        status="sent",
                        attempt=attempt
                    )

                logger.info(
                    f"[EMAIL_SERVICE] Email sent successfully",
                    extra={
                        "to": message.to,
                        "subject": message.subject,
                        "attempt": attempt
                    }
                )

                return {
                    "success": True,
                    "message_id": result.get("message_id"),
                    "recipients": message.to,
                    "attempt": attempt,
                    "sent_at": datetime.utcnow().isoformat()
                }

            except Exception as e:
                last_error = e
                log_error(
                    e,
                    context=f"EmailService.send_email (attempt {attempt}/{max_retries})"
                )

                # Exponential backoff before retry
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # 2s, 4s, 8s
                    logger.info(f"[EMAIL_SERVICE] Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All retries failed
        if self.supabase and org_id:
            self._track_delivery(
                org_id=org_id,
                message=message,
                status="failed",
                error=str(last_error),
                attempt=max_retries
            )

        return {
            "success": False,
            "error": str(last_error),
            "attempts": max_retries
        }

    def _send_via_smtp(self, message: EmailMessage) -> Dict[str, Any]:
        """
        Send email via SMTP.

        Args:
            message: EmailMessage to send

        Returns:
            Dict with message_id and status
        """
        # Create MIME message
        mime_message = MIMEMultipart('alternative')
        mime_message['Subject'] = message.subject
        mime_message['From'] = f"{self.config.from_name} <{self.config.from_email}>"
        mime_message['To'] = ', '.join(message.to)

        if message.cc:
            mime_message['Cc'] = ', '.join(message.cc)

        if message.reply_to:
            mime_message['Reply-To'] = message.reply_to

        # Add plain text part
        text_part = MIMEText(message.body_text, 'plain')
        mime_message.attach(text_part)

        # Add HTML part if provided
        if message.body_html:
            html_part = MIMEText(message.body_html, 'html')
            mime_message.attach(html_part)

        # Add attachments
        for attachment in message.attachments:
            self._add_attachment(mime_message, attachment)

        # Send via SMTP
        context = ssl.create_default_context()

        all_recipients = message.to + message.cc + message.bcc

        if self.config.use_ssl:
            # Use SMTP_SSL (port 465)
            with smtplib.SMTP_SSL(
                self.config.smtp_host,
                self.config.smtp_port,
                context=context
            ) as server:
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(mime_message, to_addrs=all_recipients)
        else:
            # Use SMTP with STARTTLS (port 587)
            with smtplib.SMTP(
                self.config.smtp_host,
                self.config.smtp_port
            ) as server:
                if self.config.use_tls:
                    server.starttls(context=context)
                server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(mime_message, to_addrs=all_recipients)

        # Generate message ID for tracking
        message_id = mime_message['Message-ID'] if 'Message-ID' in mime_message else None

        return {
            "message_id": message_id,
            "status": "sent"
        }

    def _add_attachment(self, mime_message: MIMEMultipart, attachment: Dict[str, Any]):
        """
        Add attachment to MIME message.

        Args:
            mime_message: MIME message to add attachment to
            attachment: Dict with 'filename' and either 'content' or 'path'
        """
        try:
            # Get attachment data
            if 'content' in attachment:
                data = attachment['content']
            elif 'path' in attachment:
                with open(attachment['path'], 'rb') as f:
                    data = f.read()
            else:
                raise ValueError("Attachment must have 'content' or 'path'")

            # Create attachment part
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(data)
            encoders.encode_base64(part)

            # Add header
            filename = attachment.get('filename', 'attachment')
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={filename}'
            )

            mime_message.attach(part)

        except Exception as e:
            log_error(e, context="EmailService._add_attachment")
            # Don't fail entire email if attachment fails
            logger.warning(f"[EMAIL_SERVICE] Failed to add attachment: {e}")

    def _track_delivery(
        self,
        org_id: str,
        message: EmailMessage,
        status: str,
        error: Optional[str] = None,
        attempt: int = 1
    ):
        """
        Track email delivery in database.

        Args:
            org_id: Organization ID
            message: Email message
            status: Delivery status (sent/failed)
            error: Error message if failed
            attempt: Attempt number
        """
        try:
            delivery_record = {
                "org_id": org_id,
                "recipients": message.to,
                "subject": message.subject,
                "status": status,
                "error": error,
                "attempt": attempt,
                "sent_at": datetime.utcnow().isoformat()
            }

            self.supabase.table("email_deliveries")\
                .insert(delivery_record)\
                .execute()

        except Exception as e:
            # Don't fail email send if tracking fails
            log_error(e, context="EmailService._track_delivery")

    def send_campaign_summary(
        self,
        to: List[str],
        campaign_name: str,
        metrics: Dict[str, Any],
        org_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a marketing campaign summary email.

        Args:
            to: List of recipient email addresses
            campaign_name: Name of the campaign
            metrics: Campaign metrics
            org_id: Organization ID

        Returns:
            Delivery result
        """
        subject = f"Campaign Summary: {campaign_name}"

        # Plain text version
        body_text = f"""
Campaign Summary: {campaign_name}

Results:
- Total Actions: {metrics.get('total_actions', 0)}
- Successful: {metrics.get('successful_actions', 0)}
- Messages Sent: {metrics.get('messages_sent', 0)}
- Tasks Created: {metrics.get('tasks_created', 0)}

Completed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

---
Vibodh AI - Your AI Marketing Assistant
        """.strip()

        # HTML version
        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background: #4CAF50; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; }}
        .metrics {{ background: #f4f4f4; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ margin: 10px 0; }}
        .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎉 Campaign Summary</h1>
        <h2>{campaign_name}</h2>
    </div>

    <div class="content">
        <div class="metrics">
            <h3>Results</h3>
            <div class="metric">📊 Total Actions: <strong>{metrics.get('total_actions', 0)}</strong></div>
            <div class="metric">✅ Successful: <strong>{metrics.get('successful_actions', 0)}</strong></div>
            <div class="metric">💬 Messages Sent: <strong>{metrics.get('messages_sent', 0)}</strong></div>
            <div class="metric">📋 Tasks Created: <strong>{metrics.get('tasks_created', 0)}</strong></div>
        </div>

        <p>Completed at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
    </div>

    <div class="footer">
        <p>Vibodh AI - Your AI Marketing Assistant</p>
    </div>
</body>
</html>
        """.strip()

        message = EmailMessage(
            to=to,
            subject=subject,
            body_text=body_text,
            body_html=body_html
        )

        return self.send_email(message, org_id=org_id)

    def send_notification(
        self,
        to: List[str],
        title: str,
        message_text: str,
        org_id: Optional[str] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Send a system notification email.

        Args:
            to: List of recipient email addresses
            title: Notification title
            message_text: Notification message
            org_id: Organization ID
            priority: Priority level (low/normal/high/critical)

        Returns:
            Delivery result
        """
        priority_emoji = {
            "low": "ℹ️",
            "normal": "📢",
            "high": "⚠️",
            "critical": "🚨"
        }

        emoji = priority_emoji.get(priority, "📢")
        subject = f"{emoji} {title}"

        body_text = f"""
{title}

{message_text}

---
Vibodh AI System Notification
        """.strip()

        body_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .header {{ background: #2196F3; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h2>{emoji} {title}</h2>
    </div>

    <div class="content">
        <p>{message_text}</p>
    </div>

    <div class="footer">
        <p>Vibodh AI System Notification</p>
    </div>
</body>
</html>
        """.strip()

        message = EmailMessage(
            to=to,
            subject=subject,
            body_text=body_text,
            body_html=body_html
        )

        return self.send_email(message, org_id=org_id)


# Global instance (lazy initialization)
_email_service_instance = None


def get_email_service(config: Optional[EmailConfig] = None, supabase=None) -> EmailService:
    """
    Get or create the global EmailService instance.

    Args:
        config: Email configuration (only used on first call)
        supabase: Supabase client

    Returns:
        EmailService instance
    """
    global _email_service_instance

    if _email_service_instance is None:
        if config is None:
            # Try to load from settings
            config = EmailConfig(
                smtp_host=getattr(settings, 'SMTP_HOST', ''),
                smtp_port=getattr(settings, 'SMTP_PORT', 587),
                smtp_username=getattr(settings, 'SMTP_USERNAME', ''),
                smtp_password=getattr(settings, 'SMTP_PASSWORD', ''),
                from_email=getattr(settings, 'FROM_EMAIL', ''),
                from_name=getattr(settings, 'FROM_NAME', 'Vibodh AI'),
                use_tls=getattr(settings, 'SMTP_USE_TLS', True),
                use_ssl=getattr(settings, 'SMTP_USE_SSL', False)
            )

        _email_service_instance = EmailService(config, supabase)

    return _email_service_instance
