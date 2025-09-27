"""
Alert Management System for FraudGuard 360
Handles fraud alerts, notifications, and case management
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, asdict
import sqlite3
import aiosqlite
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"

class AlertType(str, Enum):
    FRAUD_DETECTION = "fraud_detection"
    VELOCITY_FRAUD = "velocity_fraud"
    SIM_BOX_FRAUD = "sim_box_fraud"
    PREMIUM_RATE_FRAUD = "premium_rate_fraud"
    ACCOUNT_TAKEOVER = "account_takeover"
    ROAMING_FRAUD = "roaming_fraud"
    LOCATION_ANOMALY = "location_anomaly"
    SYSTEM_ALERT = "system_alert"

class FraudAlert(BaseModel):
    """Fraud alert model"""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.OPEN
    title: str
    description: str
    user_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    risk_score: float = Field(ge=0.0, le=1.0)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    location: Optional[str] = None
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    escalation_level: int = Field(default=0, ge=0, le=3)
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AlertUpdate(BaseModel):
    """Alert update model"""
    status: Optional[AlertStatus] = None
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    escalation_level: Optional[int] = None
    tags: Optional[List[str]] = None

class AlertQuery(BaseModel):
    """Alert query filters"""
    alert_type: Optional[AlertType] = None
    severity: Optional[AlertSeverity] = None
    status: Optional[AlertStatus] = None
    user_id: Optional[str] = None
    assigned_to: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)

class AlertStats(BaseModel):
    """Alert statistics"""
    total_alerts: int
    open_alerts: int
    resolved_alerts: int
    high_severity_alerts: int
    alerts_by_type: Dict[str, int]
    alerts_by_status: Dict[str, int]
    avg_resolution_time_hours: float
    alert_trend_24h: List[int]

@dataclass
class WebSocketConnection:
    """WebSocket connection data"""
    websocket: WebSocket
    user_id: str
    subscribed_types: Set[AlertType]
    last_ping: datetime

class AlertManager:
    """Main alert management service"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.alert_cache: Dict[str, FraudAlert] = {}
        self.notification_rules: Dict[AlertType, Dict[str, Any]] = {}
        self._setup_notification_rules()
    
    async def initialize_database(self):
        """Initialize SQLite database for alert storage"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    evidence TEXT,
                    location TEXT,
                    assigned_to TEXT,
                    resolved_at TEXT,
                    resolution_notes TEXT,
                    escalation_level INTEGER DEFAULT 0,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id)
            """)
            
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)
            """)
            
            await db.commit()
    
    def _setup_notification_rules(self):
        """Setup notification rules for different alert types"""
        self.notification_rules = {
            AlertType.FRAUD_DETECTION: {
                "auto_escalate_after_minutes": 30,
                "notify_channels": ["email", "sms", "dashboard"]
            },
            AlertType.SIM_BOX_FRAUD: {
                "auto_escalate_after_minutes": 15,
                "notify_channels": ["email", "sms", "dashboard", "slack"]
            },
            AlertType.ACCOUNT_TAKEOVER: {
                "auto_escalate_after_minutes": 10,
                "notify_channels": ["email", "sms", "dashboard", "slack", "pager"]
            },
            AlertType.SYSTEM_ALERT: {
                "auto_escalate_after_minutes": 60,
                "notify_channels": ["email", "dashboard"]
            }
        }
    
    async def create_alert(self, alert: FraudAlert) -> FraudAlert:
        """Create a new fraud alert"""
        try:
            # Store in database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO alerts (
                        alert_id, alert_type, severity, status, title, description,
                        user_id, timestamp, risk_score, evidence, location,
                        assigned_to, resolved_at, resolution_notes, escalation_level, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.alert_id, alert.alert_type.value, alert.severity.value,
                    alert.status.value, alert.title, alert.description, alert.user_id,
                    alert.timestamp.isoformat(), alert.risk_score,
                    json.dumps(alert.evidence), alert.location, alert.assigned_to,
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.resolution_notes, alert.escalation_level,
                    json.dumps(alert.tags)
                ))
                await db.commit()
            
            # Cache alert
            self.alert_cache[alert.alert_id] = alert
            
            # Notify connected clients
            await self._broadcast_alert(alert)
            
            # Apply notification rules
            await self._apply_notification_rules(alert)
            
            logger.info(f"Created alert {alert.alert_id}: {alert.title}")
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise HTTPException(status_code=500, detail="Failed to create alert")
    
    async def update_alert(self, alert_id: str, update: AlertUpdate) -> Optional[FraudAlert]:
        """Update an existing alert"""
        try:
            # Get current alert
            alert = await self.get_alert(alert_id)
            if not alert:
                return None
            
            # Apply updates
            update_data = update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(alert, field, value)
            
            # Set resolved timestamp if status changed to resolved
            if update.status and update.status in [AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE]:
                alert.resolved_at = datetime.now(timezone.utc)
            
            # Update in database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE alerts SET
                        status = ?, assigned_to = ?, resolved_at = ?,
                        resolution_notes = ?, escalation_level = ?, tags = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE alert_id = ?
                """, (
                    alert.status.value, alert.assigned_to,
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.resolution_notes, alert.escalation_level,
                    json.dumps(alert.tags), alert_id
                ))
                await db.commit()
            
            # Update cache
            self.alert_cache[alert_id] = alert
            
            # Broadcast update
            await self._broadcast_alert_update(alert)
            
            logger.info(f"Updated alert {alert_id}")
            return alert
            
        except Exception as e:
            logger.error(f"Failed to update alert {alert_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update alert")
    
    async def get_alert(self, alert_id: str) -> Optional[FraudAlert]:
        """Get a specific alert"""
        # Check cache first
        if alert_id in self.alert_cache:
            return self.alert_cache[alert_id]
        
        # Query database
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM alerts WHERE alert_id = ?
                """, (alert_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        alert = self._row_to_alert(row)
                        self.alert_cache[alert_id] = alert
                        return alert
        except Exception as e:
            logger.error(f"Failed to get alert {alert_id}: {e}")
        
        return None
    
    async def query_alerts(self, query: AlertQuery) -> List[FraudAlert]:
        """Query alerts with filters"""
        try:
            conditions = []
            params = []
            
            if query.alert_type:
                conditions.append("alert_type = ?")
                params.append(query.alert_type.value)
            
            if query.severity:
                conditions.append("severity = ?")
                params.append(query.severity.value)
            
            if query.status:
                conditions.append("status = ?")
                params.append(query.status.value)
            
            if query.user_id:
                conditions.append("user_id = ?")
                params.append(query.user_id)
            
            if query.assigned_to:
                conditions.append("assigned_to = ?")
                params.append(query.assigned_to)
            
            if query.start_date:
                conditions.append("timestamp >= ?")
                params.append(query.start_date.isoformat())
            
            if query.end_date:
                conditions.append("timestamp <= ?")
                params.append(query.end_date.isoformat())
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            sql = f"""
                SELECT * FROM alerts {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            params.extend([query.limit, query.offset])
            
            alerts = []
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        alerts.append(self._row_to_alert(row))
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to query alerts: {e}")
            return []
    
    async def get_alert_statistics(self) -> AlertStats:
        """Get alert statistics and metrics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Total alerts
                async with db.execute("SELECT COUNT(*) FROM alerts") as cursor:
                    total_alerts = (await cursor.fetchone())[0]
                
                # Open alerts
                async with db.execute("SELECT COUNT(*) FROM alerts WHERE status = 'open'") as cursor:
                    open_alerts = (await cursor.fetchone())[0]
                
                # Resolved alerts
                async with db.execute("SELECT COUNT(*) FROM alerts WHERE status = 'resolved'") as cursor:
                    resolved_alerts = (await cursor.fetchone())[0]
                
                # High severity alerts
                async with db.execute("SELECT COUNT(*) FROM alerts WHERE severity IN ('high', 'critical')") as cursor:
                    high_severity_alerts = (await cursor.fetchone())[0]
                
                # Alerts by type
                alerts_by_type = {}
                async with db.execute("SELECT alert_type, COUNT(*) FROM alerts GROUP BY alert_type") as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        alerts_by_type[row[0]] = row[1]
                
                # Alerts by status
                alerts_by_status = {}
                async with db.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status") as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        alerts_by_status[row[0]] = row[1]
                
                # Average resolution time
                avg_resolution_time = 0.0
                async with db.execute("""
                    SELECT AVG(
                        (julianday(resolved_at) - julianday(timestamp)) * 24
                    ) FROM alerts WHERE resolved_at IS NOT NULL
                """) as cursor:
                    result = await cursor.fetchone()
                    avg_resolution_time = result[0] if result[0] else 0.0
                
                # 24h trend (hourly counts)
                alert_trend_24h = []
                for i in range(24):
                    hour_start = datetime.now(timezone.utc) - timedelta(hours=23-i)
                    hour_end = hour_start + timedelta(hours=1)
                    async with db.execute("""
                        SELECT COUNT(*) FROM alerts 
                        WHERE timestamp >= ? AND timestamp < ?
                    """, (hour_start.isoformat(), hour_end.isoformat())) as cursor:
                        count = (await cursor.fetchone())[0]
                        alert_trend_24h.append(count)
                
                return AlertStats(
                    total_alerts=total_alerts,
                    open_alerts=open_alerts,
                    resolved_alerts=resolved_alerts,
                    high_severity_alerts=high_severity_alerts,
                    alerts_by_type=alerts_by_type,
                    alerts_by_status=alerts_by_status,
                    avg_resolution_time_hours=avg_resolution_time,
                    alert_trend_24h=alert_trend_24h
                )
                
        except Exception as e:
            logger.error(f"Failed to get alert statistics: {e}")
            return AlertStats(
                total_alerts=0, open_alerts=0, resolved_alerts=0,
                high_severity_alerts=0, alerts_by_type={}, alerts_by_status={},
                avg_resolution_time_hours=0.0, alert_trend_24h=[0] * 24
            )
    
    async def connect_websocket(self, websocket: WebSocket, user_id: str, 
                               subscribed_types: List[AlertType] = None):
        """Connect a WebSocket client for real-time alerts"""
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(
            websocket=websocket,
            user_id=user_id,
            subscribed_types=set(subscribed_types or []),
            last_ping=datetime.now(timezone.utc)
        )
        
        self.active_connections[connection_id] = connection
        logger.info(f"WebSocket connected: {user_id} ({connection_id})")
        
        try:
            # Send initial alert count
            stats = await self.get_alert_statistics()
            await websocket.send_json({
                "type": "stats_update",
                "data": stats.dict()
            })
            
            # Keep connection alive
            while True:
                await asyncio.sleep(30)  # Ping every 30 seconds
                await websocket.send_json({"type": "ping"})
                connection.last_ping = datetime.now(timezone.utc)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {user_id}: {e}")
        finally:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
    
    async def _broadcast_alert(self, alert: FraudAlert):
        """Broadcast new alert to connected WebSocket clients"""
        message = {
            "type": "new_alert",
            "data": alert.dict()
        }
        
        disconnected = []
        for conn_id, conn in self.active_connections.items():
            try:
                if not conn.subscribed_types or alert.alert_type in conn.subscribed_types:
                    await conn.websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send alert to {conn.user_id}: {e}")
                disconnected.append(conn_id)
        
        # Clean up disconnected clients
        for conn_id in disconnected:
            del self.active_connections[conn_id]
    
    async def _broadcast_alert_update(self, alert: FraudAlert):
        """Broadcast alert update to connected clients"""
        message = {
            "type": "alert_update",
            "data": alert.dict()
        }
        
        disconnected = []
        for conn_id, conn in self.active_connections.items():
            try:
                await conn.websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send update to {conn.user_id}: {e}")
                disconnected.append(conn_id)
        
        # Clean up disconnected clients  
        for conn_id in disconnected:
            del self.active_connections[conn_id]
    
    async def _apply_notification_rules(self, alert: FraudAlert):
        """Apply notification rules for the alert"""
        rules = self.notification_rules.get(alert.alert_type, {})
        
        # Schedule auto-escalation
        escalate_after = rules.get("auto_escalate_after_minutes", 60)
        if escalate_after > 0:
            asyncio.create_task(self._auto_escalate_alert(alert.alert_id, escalate_after))
        
        # Log notification channels (in real implementation, send actual notifications)
        channels = rules.get("notify_channels", [])
        logger.info(f"Alert {alert.alert_id} notifications sent to: {channels}")
    
    async def _auto_escalate_alert(self, alert_id: str, delay_minutes: int):
        """Auto-escalate alert after specified delay"""
        await asyncio.sleep(delay_minutes * 60)
        
        alert = await self.get_alert(alert_id)
        if alert and alert.status == AlertStatus.OPEN:
            await self.update_alert(alert_id, AlertUpdate(
                escalation_level=min(alert.escalation_level + 1, 3)
            ))
            logger.info(f"Auto-escalated alert {alert_id} to level {alert.escalation_level + 1}")
    
    def _row_to_alert(self, row) -> FraudAlert:
        """Convert database row to FraudAlert object"""
        return FraudAlert(
            alert_id=row[0],
            alert_type=AlertType(row[1]),
            severity=AlertSeverity(row[2]),
            status=AlertStatus(row[3]),
            title=row[4],
            description=row[5],
            user_id=row[6],
            timestamp=datetime.fromisoformat(row[7]),
            risk_score=row[8],
            evidence=json.loads(row[9]) if row[9] else {},
            location=row[10],
            assigned_to=row[11],
            resolved_at=datetime.fromisoformat(row[12]) if row[12] else None,
            resolution_notes=row[13],
            escalation_level=row[14] or 0,
            tags=json.loads(row[15]) if row[15] else []
        )

# FastAPI Application
app = FastAPI(
    title="FraudGuard 360 Alert Management",
    description="Fraud alert management and notification service",
    version="1.0.0"
)

# Global alert manager
alert_manager = AlertManager("alerts.db")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await alert_manager.initialize_database()
    logger.info("Alert Management Service started")

@app.post("/api/v1/alerts", response_model=FraudAlert)
async def create_alert(alert: FraudAlert):
    """Create a new fraud alert"""
    return await alert_manager.create_alert(alert)

@app.get("/api/v1/alerts/{alert_id}", response_model=FraudAlert)
async def get_alert(alert_id: str):
    """Get a specific alert"""
    alert = await alert_manager.get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

@app.put("/api/v1/alerts/{alert_id}", response_model=FraudAlert)
async def update_alert(alert_id: str, update: AlertUpdate):
    """Update an existing alert"""
    alert = await alert_manager.update_alert(alert_id, update)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert

@app.post("/api/v1/alerts/query", response_model=List[FraudAlert])
async def query_alerts(query: AlertQuery):
    """Query alerts with filters"""
    return await alert_manager.query_alerts(query)

@app.get("/api/v1/alerts/stats", response_model=AlertStats)
async def get_alert_statistics():
    """Get alert statistics and metrics"""
    return await alert_manager.get_alert_statistics()

@app.websocket("/ws/alerts/{user_id}")
async def websocket_alerts(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time alerts"""
    await websocket.accept()
    await alert_manager.connect_websocket(websocket, user_id)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_connections": len(alert_manager.active_connections)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)