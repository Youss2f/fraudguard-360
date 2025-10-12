"""
Notification System for FraudGuard 360
Handles multi-channel notifications, user preferences, and delivery tracking
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiofiles
import httpx
import uuid
import sqlite3
import aiosqlite
from jinja2 import Template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"
    IN_APP = "in_app"
    PUSH = "push"

class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"

class NotificationType(str, Enum):
    FRAUD_ALERT = "fraud_alert"
    SYSTEM_ALERT = "system_alert"
    CASE_UPDATE = "case_update"
    REPORT_READY = "report_ready"
    MAINTENANCE = "maintenance"
    SECURITY = "security"
    MARKETING = "marketing"

class NotificationTemplate(BaseModel):
    """Notification template model"""
    template_id: str
    name: str
    type: NotificationType
    channel: NotificationChannel
    subject_template: str
    content_template: str
    variables: List[str] = Field(default_factory=list)
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NotificationPreference(BaseModel):
    """User notification preferences"""
    user_id: str
    notification_type: NotificationType
    channels: List[NotificationChannel]
    enabled: bool = True
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None
    max_frequency_per_hour: int = Field(default=10, ge=1, le=100)

class NotificationRecipient(BaseModel):
    """Notification recipient"""
    user_id: str
    name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    slack_user_id: Optional[str] = None
    webhook_url: Optional[str] = None
    timezone: str = "UTC"

class NotificationRequest(BaseModel):
    """Notification request model"""
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: NotificationType
    priority: NotificationPriority = NotificationPriority.NORMAL
    recipients: List[NotificationRecipient]
    channels: List[NotificationChannel]
    subject: str
    content: str
    template_variables: Dict[str, Any] = Field(default_factory=dict)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    attachments: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NotificationDelivery(BaseModel):
    """Notification delivery record"""
    delivery_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    notification_id: str
    recipient_user_id: str
    channel: NotificationChannel
    status: NotificationStatus = NotificationStatus.PENDING
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    next_retry_at: Optional[datetime] = None

class NotificationStats(BaseModel):
    """Notification statistics"""
    total_sent: int
    total_delivered: int
    total_failed: int
    delivery_rate: float
    avg_delivery_time_seconds: float
    stats_by_channel: Dict[str, Dict[str, int]]
    stats_by_type: Dict[str, Dict[str, int]]
    recent_failures: List[str]

class EmailConfig(BaseModel):
    """Email configuration"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str
    password: str
    use_tls: bool = True
    sender_name: str = "FraudGuard 360"
    sender_email: str

class SlackConfig(BaseModel):
    """Slack configuration"""
    bot_token: str
    webhook_url: Optional[str] = None

class SMSConfig(BaseModel):
    """SMS configuration"""
    api_key: str
    api_secret: str
    service_provider: str = "twilio"  # twilio, nexmo, etc.

class NotificationManager:
    """Main notification management service"""
    
    def __init__(self, db_path: str = "notifications.db"):
        self.db_path = db_path
        self.templates: Dict[str, NotificationTemplate] = {}
        self.user_preferences: Dict[str, List[NotificationPreference]] = {}
        self.email_config: Optional[EmailConfig] = None
        self.slack_config: Optional[SlackConfig] = None
        self.sms_config: Optional[SMSConfig] = None
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.worker_tasks: List[asyncio.Task] = []
        self._setup_default_templates()
    
    async def initialize_database(self):
        """Initialize notification database"""
        async with aiosqlite.connect(self.db_path) as db:
            # Notification templates table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS notification_templates (
                    template_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    subject_template TEXT NOT NULL,
                    content_template TEXT NOT NULL,
                    variables TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TEXT NOT NULL
                )
            """)
            
            # User preferences table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS notification_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    notification_type TEXT NOT NULL,
                    channels TEXT NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    quiet_hours_start TEXT,
                    quiet_hours_end TEXT,
                    max_frequency_per_hour INTEGER DEFAULT 10,
                    UNIQUE(user_id, notification_type)
                )
            """)
            
            # Notification requests table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS notification_requests (
                    notification_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    template_variables TEXT,
                    scheduled_at TEXT,
                    expires_at TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Notification deliveries table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS notification_deliveries (
                    delivery_id TEXT PRIMARY KEY,
                    notification_id TEXT NOT NULL,
                    recipient_user_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    status TEXT NOT NULL,
                    sent_at TEXT,
                    delivered_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    next_retry_at TEXT,
                    FOREIGN KEY (notification_id) REFERENCES notification_requests (notification_id)
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_status ON notification_deliveries(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_scheduled ON notification_deliveries(next_retry_at)")
            
            await db.commit()
    
    def configure_email(self, config: EmailConfig):
        """Configure email settings"""
        self.email_config = config
        logger.info("Email configuration updated")
    
    def configure_slack(self, config: SlackConfig):
        """Configure Slack settings"""
        self.slack_config = config
        logger.info("Slack configuration updated")
    
    def configure_sms(self, config: SMSConfig):
        """Configure SMS settings"""
        self.sms_config = config
        logger.info("SMS configuration updated")
    
    async def start_workers(self, num_workers: int = 3):
        """Start notification delivery workers"""
        for i in range(num_workers):
            task = asyncio.create_task(self._delivery_worker(f"worker-{i}"))
            self.worker_tasks.append(task)
        logger.info(f"Started {num_workers} notification workers")
    
    async def stop_workers(self):
        """Stop all notification workers"""
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("All notification workers stopped")
    
    async def send_notification(self, request: NotificationRequest) -> List[NotificationDelivery]:
        """Send notification to all recipients"""
        try:
            # Store notification request
            await self._store_notification_request(request)
            
            deliveries = []
            
            for recipient in request.recipients:
                # Check user preferences
                allowed_channels = await self._get_allowed_channels(
                    recipient.user_id, request.type, request.channels
                )
                
                # Create delivery records for each allowed channel
                for channel in allowed_channels:
                    delivery = NotificationDelivery(
                        notification_id=request.notification_id,
                        recipient_user_id=recipient.user_id,
                        channel=channel
                    )
                    
                    deliveries.append(delivery)
                    await self._store_delivery_record(delivery)
                    
                    # Queue for delivery
                    await self.delivery_queue.put((request, recipient, delivery))
            
            logger.info(f"Queued notification {request.notification_id} for {len(deliveries)} deliveries")
            return deliveries
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            raise HTTPException(status_code=500, detail="Failed to send notification")
    
    async def send_template_notification(
        self, 
        template_id: str,
        recipients: List[NotificationRecipient],
        variables: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> List[NotificationDelivery]:
        """Send notification using a template"""
        
        template = self.templates.get(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Render template
        subject = Template(template.subject_template).render(**variables)
        content = Template(template.content_template).render(**variables)
        
        request = NotificationRequest(
            type=template.type,
            priority=priority,
            recipients=recipients,
            channels=[template.channel],
            subject=subject,
            content=content,
            template_variables=variables
        )
        
        return await self.send_notification(request)
    
    async def get_delivery_status(self, notification_id: str) -> List[NotificationDelivery]:
        """Get delivery status for a notification"""
        try:
            deliveries = []
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM notification_deliveries 
                    WHERE notification_id = ?
                """, (notification_id,)) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        deliveries.append(self._row_to_delivery(row))
            
            return deliveries
            
        except Exception as e:
            logger.error(f"Failed to get delivery status: {e}")
            return []
    
    async def get_notification_stats(self, days: int = 7) -> NotificationStats:
        """Get notification statistics"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                # Total counts
                total_sent = await self._get_count(db, """
                    SELECT COUNT(*) FROM notification_deliveries 
                    WHERE sent_at >= ?
                """, (cutoff_date.isoformat(),))
                
                total_delivered = await self._get_count(db, """
                    SELECT COUNT(*) FROM notification_deliveries 
                    WHERE delivered_at >= ? AND status = 'delivered'
                """, (cutoff_date.isoformat(),))
                
                total_failed = await self._get_count(db, """
                    SELECT COUNT(*) FROM notification_deliveries 
                    WHERE sent_at >= ? AND status = 'failed'
                """, (cutoff_date.isoformat(),))
                
                # Delivery rate
                delivery_rate = total_delivered / total_sent if total_sent > 0 else 0.0
                
                # Average delivery time
                avg_delivery_time = 0.0
                async with db.execute("""
                    SELECT AVG((julianday(delivered_at) - julianday(sent_at)) * 86400)
                    FROM notification_deliveries 
                    WHERE delivered_at >= ? AND sent_at IS NOT NULL AND delivered_at IS NOT NULL
                """, (cutoff_date.isoformat(),)) as cursor:
                    result = await cursor.fetchone()
                    avg_delivery_time = result[0] if result[0] else 0.0
                
                # Stats by channel
                stats_by_channel = {}
                async with db.execute("""
                    SELECT channel, status, COUNT(*) 
                    FROM notification_deliveries 
                    WHERE sent_at >= ? 
                    GROUP BY channel, status
                """, (cutoff_date.isoformat(),)) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        channel, status, count = row
                        if channel not in stats_by_channel:
                            stats_by_channel[channel] = {}
                        stats_by_channel[channel][status] = count
                
                # Stats by type
                stats_by_type = {}
                async with db.execute("""
                    SELECT nr.type, nd.status, COUNT(*) 
                    FROM notification_deliveries nd
                    JOIN notification_requests nr ON nd.notification_id = nr.notification_id
                    WHERE nd.sent_at >= ? 
                    GROUP BY nr.type, nd.status
                """, (cutoff_date.isoformat(),)) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        notification_type, status, count = row
                        if notification_type not in stats_by_type:
                            stats_by_type[notification_type] = {}
                        stats_by_type[notification_type][status] = count
                
                # Recent failures
                recent_failures = []
                async with db.execute("""
                    SELECT error_message FROM notification_deliveries 
                    WHERE status = 'failed' AND sent_at >= ?
                    ORDER BY sent_at DESC LIMIT 10
                """, (cutoff_date.isoformat(),)) as cursor:
                    rows = await cursor.fetchall()
                    recent_failures = [row[0] for row in rows if row[0]]
                
                return NotificationStats(
                    total_sent=total_sent,
                    total_delivered=total_delivered,
                    total_failed=total_failed,
                    delivery_rate=delivery_rate,
                    avg_delivery_time_seconds=avg_delivery_time,
                    stats_by_channel=stats_by_channel,
                    stats_by_type=stats_by_type,
                    recent_failures=recent_failures
                )
                
        except Exception as e:
            logger.error(f"Failed to get notification stats: {e}")
            return NotificationStats(
                total_sent=0, total_delivered=0, total_failed=0,
                delivery_rate=0.0, avg_delivery_time_seconds=0.0,
                stats_by_channel={}, stats_by_type={}, recent_failures=[]
            )
    
    async def _delivery_worker(self, worker_name: str):
        """Worker to process notification deliveries"""
        logger.info(f"Notification worker {worker_name} started")
        
        while True:
            try:
                # Get delivery task from queue
                request, recipient, delivery = await self.delivery_queue.get()
                
                # Check if notification has expired
                if request.expires_at and datetime.now(timezone.utc) > request.expires_at:
                    delivery.status = NotificationStatus.FAILED
                    delivery.error_message = "Notification expired"
                    await self._update_delivery_status(delivery)
                    continue
                
                # Process delivery based on channel
                success = await self._deliver_notification(request, recipient, delivery)
                
                if success:
                    delivery.status = NotificationStatus.SENT
                    delivery.sent_at = datetime.now(timezone.utc)
                else:
                    delivery.retry_count += 1
                    if delivery.retry_count >= delivery.max_retries:
                        delivery.status = NotificationStatus.FAILED
                    else:
                        delivery.status = NotificationStatus.RETRYING
                        # Exponential backoff
                        delay_minutes = 2 ** delivery.retry_count
                        delivery.next_retry_at = datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)
                        
                        # Re-queue for retry
                        await asyncio.sleep(delay_minutes * 60)
                        await self.delivery_queue.put((request, recipient, delivery))
                
                await self._update_delivery_status(delivery)
                
            except asyncio.CancelledError:
                logger.info(f"Notification worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in notification worker {worker_name}: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _deliver_notification(
        self, 
        request: NotificationRequest,
        recipient: NotificationRecipient,
        delivery: NotificationDelivery
    ) -> bool:
        """Deliver notification via specific channel"""
        try:
            if delivery.channel == NotificationChannel.EMAIL:
                return await self._send_email(request, recipient, delivery)
            elif delivery.channel == NotificationChannel.SLACK:
                return await self._send_slack(request, recipient, delivery)
            elif delivery.channel == NotificationChannel.SMS:
                return await self._send_sms(request, recipient, delivery)
            elif delivery.channel == NotificationChannel.WEBHOOK:
                return await self._send_webhook(request, recipient, delivery)
            else:
                logger.warning(f"Unsupported notification channel: {delivery.channel}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deliver notification via {delivery.channel}: {e}")
            delivery.error_message = str(e)
            return False
    
    async def _send_email(
        self, 
        request: NotificationRequest,
        recipient: NotificationRecipient,
        delivery: NotificationDelivery
    ) -> bool:
        """Send email notification"""
        if not self.email_config or not recipient.email:
            delivery.error_message = "Email not configured or recipient email missing"
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{self.email_config.sender_name} <{self.email_config.sender_email}>"
            msg['To'] = recipient.email
            msg['Subject'] = request.subject
            
            # Add body
            msg.attach(MIMEText(request.content, 'html' if '<' in request.content else 'plain'))
            
            # Add attachments
            for attachment_path in request.attachments:
                try:
                    async with aiofiles.open(attachment_path, 'rb') as f:
                        attachment_data = await f.read()
                    
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment_data)
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment_path.split("/")[-1]}'
                    )
                    msg.attach(part)
                except Exception as e:
                    logger.warning(f"Failed to attach file {attachment_path}: {e}")
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.email_config.smtp_server, self.email_config.smtp_port) as server:
                if self.email_config.use_tls:
                    server.starttls(context=context)
                server.login(self.email_config.username, self.email_config.password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {recipient.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            delivery.error_message = str(e)
            return False
    
    async def _send_slack(
        self, 
        request: NotificationRequest,
        recipient: NotificationRecipient,
        delivery: NotificationDelivery
    ) -> bool:
        """Send Slack notification"""
        if not self.slack_config:
            delivery.error_message = "Slack not configured"
            return False
        
        try:
            # Prepare Slack message
            message = {
                "text": request.subject,
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{request.subject}*\n{request.content}"
                        }
                    }
                ]
            }
            
            # Add user mention if available
            if recipient.slack_user_id:
                message["channel"] = f"@{recipient.slack_user_id}"
            
            # Send via webhook or API
            async with httpx.AsyncClient() as client:
                if self.slack_config.webhook_url:
                    response = await client.post(self.slack_config.webhook_url, json=message)
                else:
                    headers = {"Authorization": f"Bearer {self.slack_config.bot_token}"}
                    response = await client.post(
                        "https://slack.com/api/chat.postMessage",
                        headers=headers,
                        json=message
                    )
                
                if response.status_code == 200:
                    logger.info(f"Slack message sent to {recipient.slack_user_id or 'webhook'}")
                    return True
                else:
                    delivery.error_message = f"Slack API error: {response.status_code}"
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            delivery.error_message = str(e)
            return False
    
    async def _send_sms(
        self, 
        request: NotificationRequest,
        recipient: NotificationRecipient,
        delivery: NotificationDelivery
    ) -> bool:
        """Send SMS notification"""
        if not self.sms_config or not recipient.phone:
            delivery.error_message = "SMS not configured or recipient phone missing"
            return False
        
        # SMS implementation would depend on the service provider
        # This is a placeholder for the actual SMS sending logic
        logger.info(f"SMS would be sent to {recipient.phone}: {request.subject}")
        return True
    
    async def _send_webhook(
        self, 
        request: NotificationRequest,
        recipient: NotificationRecipient,
        delivery: NotificationDelivery
    ) -> bool:
        """Send webhook notification"""
        if not recipient.webhook_url:
            delivery.error_message = "Webhook URL not configured"
            return False
        
        try:
            payload = {
                "notification_id": request.notification_id,
                "type": request.type,
                "priority": request.priority,
                "subject": request.subject,
                "content": request.content,
                "recipient": recipient.dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": request.metadata
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(recipient.webhook_url, json=payload)
                
                if response.status_code in [200, 201, 202]:
                    logger.info(f"Webhook sent to {recipient.webhook_url}")
                    return True
                else:
                    delivery.error_message = f"Webhook error: {response.status_code}"
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            delivery.error_message = str(e)
            return False
    
    def _setup_default_templates(self):
        """Setup default notification templates"""
        templates = [
            NotificationTemplate(
                template_id="fraud_alert_email",
                name="Fraud Alert Email",
                type=NotificationType.FRAUD_ALERT,
                channel=NotificationChannel.EMAIL,
                subject_template="🚨 Fraud Alert: {{ alert_type }}",
                content_template="""
                <h2>Fraud Alert Detected</h2>
                <p><strong>Alert Type:</strong> {{ alert_type }}</p>
                <p><strong>User ID:</strong> {{ user_id }}</p>
                <p><strong>Risk Score:</strong> {{ risk_score }}</p>
                <p><strong>Description:</strong> {{ description }}</p>
                <p><strong>Timestamp:</strong> {{ timestamp }}</p>
                
                <p>Please investigate this alert immediately in the FraudGuard dashboard.</p>
                """,
                variables=["alert_type", "user_id", "risk_score", "description", "timestamp"]
            ),
            NotificationTemplate(
                template_id="case_update_email",
                name="Case Update Email",
                type=NotificationType.CASE_UPDATE,
                channel=NotificationChannel.EMAIL,
                subject_template="Case Update: {{ case_number }} - {{ status }}",
                content_template="""
                <h2>Case Update Notification</h2>
                <p><strong>Case Number:</strong> {{ case_number }}</p>
                <p><strong>Title:</strong> {{ title }}</p>
                <p><strong>Status:</strong> {{ status }}</p>
                <p><strong>Assigned To:</strong> {{ assigned_to }}</p>
                <p><strong>Update:</strong> {{ update_description }}</p>
                
                <p>View full case details in the FraudGuard dashboard.</p>
                """,
                variables=["case_number", "title", "status", "assigned_to", "update_description"]
            )
        ]
        
        for template in templates:
            self.templates[template.template_id] = template
    
    async def _get_allowed_channels(
        self, 
        user_id: str, 
        notification_type: NotificationType,
        requested_channels: List[NotificationChannel]
    ) -> List[NotificationChannel]:
        """Get allowed notification channels for user"""
        # Check user preferences
        user_prefs = self.user_preferences.get(user_id, [])
        type_prefs = [p for p in user_prefs if p.notification_type == notification_type]
        
        if type_prefs:
            pref = type_prefs[0]
            if not pref.enabled:
                return []
            
            # Check quiet hours
            if pref.quiet_hours_start and pref.quiet_hours_end:
                now = datetime.now(timezone.utc).time()
                start_time = datetime.strptime(pref.quiet_hours_start, "%H:%M").time()
                end_time = datetime.strptime(pref.quiet_hours_end, "%H:%M").time()
                
                if start_time <= now <= end_time:
                    # During quiet hours, only allow urgent notifications
                    return []
            
            # Return intersection of requested and preferred channels
            return [c for c in requested_channels if c in pref.channels]
        
        # Default: allow all requested channels
        return requested_channels
    
    async def _store_notification_request(self, request: NotificationRequest):
        """Store notification request in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO notification_requests (
                    notification_id, type, priority, subject, content,
                    template_variables, scheduled_at, expires_at, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.notification_id, request.type.value, request.priority.value,
                request.subject, request.content, json.dumps(request.template_variables),
                request.scheduled_at.isoformat() if request.scheduled_at else None,
                request.expires_at.isoformat() if request.expires_at else None,
                datetime.now(timezone.utc).isoformat(), json.dumps(request.metadata)
            ))
            await db.commit()
    
    async def _store_delivery_record(self, delivery: NotificationDelivery):
        """Store delivery record in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO notification_deliveries (
                    delivery_id, notification_id, recipient_user_id, channel,
                    status, sent_at, delivered_at, error_message, retry_count,
                    max_retries, next_retry_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                delivery.delivery_id, delivery.notification_id, delivery.recipient_user_id,
                delivery.channel.value, delivery.status.value,
                delivery.sent_at.isoformat() if delivery.sent_at else None,
                delivery.delivered_at.isoformat() if delivery.delivered_at else None,
                delivery.error_message, delivery.retry_count, delivery.max_retries,
                delivery.next_retry_at.isoformat() if delivery.next_retry_at else None
            ))
            await db.commit()
    
    async def _update_delivery_status(self, delivery: NotificationDelivery):
        """Update delivery status in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE notification_deliveries SET
                    status = ?, sent_at = ?, delivered_at = ?, error_message = ?,
                    retry_count = ?, next_retry_at = ?
                WHERE delivery_id = ?
            """, (
                delivery.status.value,
                delivery.sent_at.isoformat() if delivery.sent_at else None,
                delivery.delivered_at.isoformat() if delivery.delivered_at else None,
                delivery.error_message, delivery.retry_count,
                delivery.next_retry_at.isoformat() if delivery.next_retry_at else None,
                delivery.delivery_id
            ))
            await db.commit()
    
    async def _get_count(self, db, query: str, params: tuple = ()) -> int:
        """Helper to get count from database"""
        async with db.execute(query, params) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else 0
    
    def _row_to_delivery(self, row) -> NotificationDelivery:
        """Convert database row to NotificationDelivery object"""
        return NotificationDelivery(
            delivery_id=row[0],
            notification_id=row[1],
            recipient_user_id=row[2],
            channel=NotificationChannel(row[3]),
            status=NotificationStatus(row[4]),
            sent_at=datetime.fromisoformat(row[5]) if row[5] else None,
            delivered_at=datetime.fromisoformat(row[6]) if row[6] else None,
            error_message=row[7],
            retry_count=row[8] or 0,
            max_retries=row[9] or 3,
            next_retry_at=datetime.fromisoformat(row[10]) if row[10] else None
        )

# FastAPI Application
app = FastAPI(
    title="FraudGuard 360 Notification Service",
    description="Multi-channel notification and delivery service",
    version="1.0.0"
)

# Global notification manager
notification_manager = NotificationManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await notification_manager.initialize_database()
    await notification_manager.start_workers()
    logger.info("Notification Service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await notification_manager.stop_workers()
    logger.info("Notification Service stopped")

@app.post("/api/v1/notifications/send", response_model=List[NotificationDelivery])
async def send_notification(request: NotificationRequest):
    """Send a notification"""
    return await notification_manager.send_notification(request)

@app.post("/api/v1/notifications/template/{template_id}", response_model=List[NotificationDelivery])
async def send_template_notification(
    template_id: str,
    recipients: List[NotificationRecipient],
    variables: Dict[str, Any],
    priority: NotificationPriority = NotificationPriority.NORMAL
):
    """Send notification using template"""
    return await notification_manager.send_template_notification(
        template_id, recipients, variables, priority
    )

@app.get("/api/v1/notifications/{notification_id}/status", response_model=List[NotificationDelivery])
async def get_delivery_status(notification_id: str):
    """Get delivery status for a notification"""
    return await notification_manager.get_delivery_status(notification_id)

@app.get("/api/v1/notifications/stats", response_model=NotificationStats)
async def get_notification_stats(days: int = 7):
    """Get notification statistics"""
    return await notification_manager.get_notification_stats(days)

@app.post("/api/v1/notifications/config/email")
async def configure_email(config: EmailConfig):
    """Configure email settings"""
    notification_manager.configure_email(config)
    return {"message": "Email configuration updated"}

@app.post("/api/v1/notifications/config/slack")
async def configure_slack(config: SlackConfig):
    """Configure Slack settings"""
    notification_manager.configure_slack(config)
    return {"message": "Slack configuration updated"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queue_size": notification_manager.delivery_queue.qsize(),
        "workers": len(notification_manager.worker_tasks)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)