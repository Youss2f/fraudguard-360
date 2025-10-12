"""
Advanced Monitoring and Metrics System for FraudGuard 360
Provides comprehensive system monitoring, performance metrics, and business intelligence
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
import psutil
import time
import statistics
from collections import defaultdict, deque
import sqlite3
import aiosqlite
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SystemStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class Metric(BaseModel):
    """Individual metric data point"""
    name: str
    value: float
    type: MetricType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = None

class SystemMetrics(BaseModel):
    """System performance metrics"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # CPU metrics
    cpu_usage_percent: float
    cpu_load_1m: float
    cpu_load_5m: float
    cpu_load_15m: float
    
    # Memory metrics
    memory_usage_percent: float
    memory_available_gb: float
    memory_used_gb: float
    memory_total_gb: float
    
    # Disk metrics
    disk_usage_percent: float
    disk_free_gb: float
    disk_total_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    
    # Network metrics
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    
    # Process metrics
    active_connections: int
    open_files: int
    process_count: int

class ServiceHealth(BaseModel):
    """Service health status"""
    service_name: str
    status: SystemStatus
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    uptime_seconds: Optional[float] = None
    version: Optional[str] = None
    
class BusinessMetrics(BaseModel):
    """Business intelligence metrics"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Transaction metrics
    total_transactions_24h: int
    successful_transactions_24h: int
    failed_transactions_24h: int
    
    # Fraud detection metrics
    fraud_cases_detected_24h: int
    false_positives_24h: int
    true_positives_24h: int
    fraud_detection_accuracy: float
    
    # Financial metrics
    total_transaction_value_24h: float
    blocked_fraudulent_value_24h: float
    estimated_savings_24h: float
    
    # Performance metrics  
    avg_detection_time_ms: float
    max_detection_time_ms: float
    min_detection_time_ms: float
    
    # Alert metrics
    total_alerts_24h: int
    critical_alerts_24h: int
    resolved_alerts_24h: int
    avg_resolution_time_minutes: float

class PerformanceAlert(BaseModel):
    """Performance monitoring alert"""
    alert_id: str
    alert_type: str
    level: AlertLevel
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.metrics_buffer: deque = deque(maxlen=10000)  # In-memory buffer
        self.service_endpoints = {
            "api_gateway": "http://localhost:8000/health",
            "ml_service": "http://localhost:8001/health", 
            "frontend": "http://localhost:3000",
            "neo4j": "http://localhost:7474",
            "kafka": "http://localhost:8080",
            "flink": "http://localhost:8081"
        }
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time": 5000.0,  # 5 seconds
            "error_rate": 5.0  # 5%
        }
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
    async def initialize_database(self):
        """Initialize metrics database"""
        async with aiosqlite.connect(self.db_path) as db:
            # System metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER,
                    active_connections INTEGER,
                    metrics_json TEXT
                )
            """)
            
            # Business metrics table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS business_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_transactions INTEGER,
                    fraud_detected INTEGER,
                    detection_accuracy REAL,
                    avg_response_time REAL,
                    metrics_json TEXT
                )
            """)
            
            # Service health table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS service_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    response_time REAL,
                    error_message TEXT
                )
            """)
            
            # Performance alerts table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    alert_id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_business_metrics_timestamp ON business_metrics(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_service_health_timestamp ON service_health(timestamp)")
            
            await db.commit()
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            connections = len(psutil.net_connections())
            processes = len(psutil.pids())
            
            metrics = SystemMetrics(
                cpu_usage_percent=cpu_percent,
                cpu_load_1m=load_avg[0],
                cpu_load_5m=load_avg[1],
                cpu_load_15m=load_avg[2],
                memory_usage_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
                disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                active_connections=connections,
                open_files=len(psutil.Process().open_files()),
                process_count=processes
            )
            
            # Store in database
            await self._store_system_metrics(metrics)
            
            # Check for alerts
            await self._check_system_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise
    
    async def check_service_health(self) -> List[ServiceHealth]:
        """Check health of all services"""
        health_results = []
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            for service_name, endpoint in self.service_endpoints.items():
                try:
                    start_time = time.time()
                    response = await client.get(endpoint)
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        status = SystemStatus.HEALTHY
                        error_message = None
                    else:
                        status = SystemStatus.WARNING
                        error_message = f"HTTP {response.status_code}"
                        
                except httpx.TimeoutException:
                    status = SystemStatus.CRITICAL
                    response_time = 5000.0
                    error_message = "Service timeout"
                except Exception as e:
                    status = SystemStatus.CRITICAL
                    response_time = None
                    error_message = str(e)
                
                health = ServiceHealth(
                    service_name=service_name,
                    status=status,
                    response_time_ms=response_time,
                    error_message=error_message
                )
                
                health_results.append(health)
                
                # Store in database
                await self._store_service_health(health)
        
        return health_results
    
    async def collect_business_metrics(self) -> BusinessMetrics:
        """Collect business intelligence metrics"""
        try:
            # This would typically query various services for business data
            # For now, we'll generate sample data based on stored metrics
            
            now = datetime.now(timezone.utc)
            yesterday = now - timedelta(days=1)
            
            # Query historical data for trends
            async with aiosqlite.connect(self.db_path) as db:
                # Get recent business metrics for trend calculation
                async with db.execute("""
                    SELECT metrics_json FROM business_metrics 
                    WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 24
                """, (yesterday.isoformat(),)) as cursor:
                    recent_metrics = []
                    async for row in cursor:
                        if row[0]:
                            recent_metrics.append(json.loads(row[0]))
            
            # Calculate business metrics (sample data for demonstration)
            metrics = BusinessMetrics(
                total_transactions_24h=50000 + len(recent_metrics) * 1000,
                successful_transactions_24h=48500 + len(recent_metrics) * 950,
                failed_transactions_24h=1500 + len(recent_metrics) * 50,
                fraud_cases_detected_24h=75 + len(recent_metrics) * 3,
                false_positives_24h=8 + len(recent_metrics),
                true_positives_24h=67 + len(recent_metrics) * 2,
                fraud_detection_accuracy=0.95 + (len(recent_metrics) * 0.001),
                total_transaction_value_24h=2500000.0 + len(recent_metrics) * 50000,
                blocked_fraudulent_value_24h=125000.0 + len(recent_metrics) * 2500,
                estimated_savings_24h=112500.0 + len(recent_metrics) * 2250,
                avg_detection_time_ms=287.5 + len(recent_metrics) * 5,
                max_detection_time_ms=1250.0 + len(recent_metrics) * 25,
                min_detection_time_ms=45.0 + len(recent_metrics),
                total_alerts_24h=25 + len(recent_metrics),
                critical_alerts_24h=3 + (len(recent_metrics) // 8),
                resolved_alerts_24h=22 + len(recent_metrics),
                avg_resolution_time_minutes=45.5 + len(recent_metrics) * 2
            )
            
            # Store in database
            await self._store_business_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")
            raise
    
    async def get_historical_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical metrics for dashboard"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            async with aiosqlite.connect(self.db_path) as db:
                # System metrics
                system_metrics = []
                async with db.execute("""
                    SELECT timestamp, cpu_usage, memory_usage, disk_usage 
                    FROM system_metrics 
                    WHERE timestamp >= ? ORDER BY timestamp
                """, (cutoff_time.isoformat(),)) as cursor:
                    async for row in cursor:
                        system_metrics.append({
                            "timestamp": row[0],
                            "cpu_usage": row[1],
                            "memory_usage": row[2],
                            "disk_usage": row[3]
                        })
                
                # Business metrics
                business_metrics = []
                async with db.execute("""
                    SELECT timestamp, total_transactions, fraud_detected, detection_accuracy 
                    FROM business_metrics 
                    WHERE timestamp >= ? ORDER BY timestamp
                """, (cutoff_time.isoformat(),)) as cursor:
                    async for row in cursor:
                        business_metrics.append({
                            "timestamp": row[0],
                            "total_transactions": row[1],
                            "fraud_detected": row[2],
                            "detection_accuracy": row[3]
                        })
                
                # Service health
                service_health = []
                async with db.execute("""
                    SELECT service_name, status, timestamp, response_time 
                    FROM service_health 
                    WHERE timestamp >= ? ORDER BY timestamp
                """, (cutoff_time.isoformat(),)) as cursor:
                    async for row in cursor:
                        service_health.append({
                            "service_name": row[0],
                            "status": row[1],
                            "timestamp": row[2],
                            "response_time": row[3]
                        })
            
            return {
                "system_metrics": system_metrics,
                "business_metrics": business_metrics,
                "service_health": service_health,
                "collection_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics: {e}")
            return {}
    
    async def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active performance alerts"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a performance alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].resolved_at = datetime.now(timezone.utc)
            
            # Update in database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE performance_alerts 
                    SET resolved = TRUE, resolved_at = ? 
                    WHERE alert_id = ?
                """, (self.active_alerts[alert_id].resolved_at.isoformat(), alert_id))
                await db.commit()
            
            return True
        return False
    
    async def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO system_metrics (
                    timestamp, cpu_usage, memory_usage, disk_usage,
                    network_bytes_sent, network_bytes_recv, active_connections, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(), metrics.cpu_usage_percent,
                metrics.memory_usage_percent, metrics.disk_usage_percent,
                metrics.network_bytes_sent, metrics.network_bytes_recv,
                metrics.active_connections, json.dumps(metrics.dict())
            ))
            await db.commit()
    
    async def _store_business_metrics(self, metrics: BusinessMetrics):
        """Store business metrics in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO business_metrics (
                    timestamp, total_transactions, fraud_detected, 
                    detection_accuracy, avg_response_time, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(), metrics.total_transactions_24h,
                metrics.fraud_cases_detected_24h, metrics.fraud_detection_accuracy,
                metrics.avg_detection_time_ms, json.dumps(metrics.dict())
            ))
            await db.commit()
    
    async def _store_service_health(self, health: ServiceHealth):
        """Store service health in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO service_health (
                    service_name, status, timestamp, response_time, error_message
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                health.service_name, health.status.value, health.last_check.isoformat(),
                health.response_time_ms, health.error_message
            ))
            await db.commit()
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system performance alerts"""
        alerts_to_create = []
        
        # CPU usage alert
        if metrics.cpu_usage_percent > self.alert_thresholds["cpu_usage"]:
            alert_id = f"cpu_high_{int(time.time())}"
            if alert_id not in self.active_alerts:
                alerts_to_create.append(PerformanceAlert(
                    alert_id=alert_id,
                    alert_type="system_performance",
                    level=AlertLevel.WARNING if metrics.cpu_usage_percent < 95 else AlertLevel.CRITICAL,
                    title="High CPU Usage",
                    description=f"CPU usage is {metrics.cpu_usage_percent:.1f}%",
                    metric_name="cpu_usage_percent",
                    current_value=metrics.cpu_usage_percent,
                    threshold_value=self.alert_thresholds["cpu_usage"]
                ))
        
        # Memory usage alert
        if metrics.memory_usage_percent > self.alert_thresholds["memory_usage"]:
            alert_id = f"memory_high_{int(time.time())}"
            if alert_id not in self.active_alerts:
                alerts_to_create.append(PerformanceAlert(
                    alert_id=alert_id,
                    alert_type="system_performance",
                    level=AlertLevel.WARNING if metrics.memory_usage_percent < 95 else AlertLevel.CRITICAL,
                    title="High Memory Usage",
                    description=f"Memory usage is {metrics.memory_usage_percent:.1f}%",
                    metric_name="memory_usage_percent",
                    current_value=metrics.memory_usage_percent,
                    threshold_value=self.alert_thresholds["memory_usage"]
                ))
        
        # Disk usage alert
        if metrics.disk_usage_percent > self.alert_thresholds["disk_usage"]:
            alert_id = f"disk_high_{int(time.time())}"
            if alert_id not in self.active_alerts:
                alerts_to_create.append(PerformanceAlert(
                    alert_id=alert_id,
                    alert_type="system_performance",
                    level=AlertLevel.CRITICAL,
                    title="High Disk Usage",
                    description=f"Disk usage is {metrics.disk_usage_percent:.1f}%",
                    metric_name="disk_usage_percent",
                    current_value=metrics.disk_usage_percent,
                    threshold_value=self.alert_thresholds["disk_usage"]
                ))
        
        # Store new alerts
        for alert in alerts_to_create:
            self.active_alerts[alert.alert_id] = alert
            await self._store_alert(alert)
            logger.warning(f"Performance alert created: {alert.title}")
    
    async def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO performance_alerts (
                    alert_id, alert_type, level, title, description,
                    metric_name, current_value, threshold_value, timestamp, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id, alert.alert_type, alert.level.value,
                alert.title, alert.description, alert.metric_name,
                alert.current_value, alert.threshold_value,
                alert.timestamp.isoformat(), alert.resolved
            ))
            await db.commit()

class MonitoringService:
    """Main monitoring service orchestrator"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.collection_task = None
        self.collection_interval = 60  # seconds
        
    async def start_monitoring(self):
        """Start the monitoring service"""
        await self.metrics_collector.initialize_database()
        self.collection_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring service started")
    
    async def stop_monitoring(self):
        """Stop the monitoring service"""
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring service stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while True:
            try:
                # Collect metrics
                await self.metrics_collector.collect_system_metrics()
                await self.metrics_collector.check_service_health()
                await self.metrics_collector.collect_business_metrics()
                
                logger.debug("Metrics collection completed")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            await asyncio.sleep(self.collection_interval)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            current_system = await self.metrics_collector.collect_system_metrics()
            current_business = await self.metrics_collector.collect_business_metrics()
            service_health = await self.metrics_collector.check_service_health()
            historical = await self.metrics_collector.get_historical_metrics(24)
            active_alerts = await self.metrics_collector.get_active_alerts()
            
            return {
                "current_system_metrics": current_system.dict(),
                "current_business_metrics": current_business.dict(),
                "service_health": [h.dict() for h in service_health],
                "historical_metrics": historical,
                "active_alerts": [a.dict() for a in active_alerts],
                "overall_status": self._calculate_overall_status(service_health, active_alerts),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    def _calculate_overall_status(self, service_health: List[ServiceHealth], 
                                active_alerts: List[PerformanceAlert]) -> str:
        """Calculate overall system status"""
        critical_alerts = [a for a in active_alerts if a.level == AlertLevel.CRITICAL]
        critical_services = [s for s in service_health if s.status == SystemStatus.CRITICAL]
        
        if critical_alerts or critical_services:
            return "critical"
        
        warning_alerts = [a for a in active_alerts if a.level == AlertLevel.WARNING]
        warning_services = [s for s in service_health if s.status == SystemStatus.WARNING]
        
        if warning_alerts or warning_services:
            return "warning"
        
        return "healthy"

# FastAPI Application
app = FastAPI(
    title="FraudGuard 360 Monitoring Service",
    description="Comprehensive system monitoring and metrics service",
    version="1.0.0"
)

# Global monitoring service
monitoring_service = MonitoringService()

@app.on_event("startup")
async def startup_event():
    """Start monitoring on application startup"""
    await monitoring_service.start_monitoring()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop monitoring on application shutdown"""
    await monitoring_service.stop_monitoring()

@app.get("/api/v1/metrics/system", response_model=SystemMetrics)
async def get_current_system_metrics():
    """Get current system metrics"""
    return await monitoring_service.metrics_collector.collect_system_metrics()

@app.get("/api/v1/metrics/business", response_model=BusinessMetrics)
async def get_current_business_metrics():
    """Get current business metrics"""
    return await monitoring_service.metrics_collector.collect_business_metrics()

@app.get("/api/v1/health/services", response_model=List[ServiceHealth])
async def get_service_health():
    """Get health status of all services"""
    return await monitoring_service.metrics_collector.check_service_health()

@app.get("/api/v1/metrics/historical")
async def get_historical_metrics(hours: int = 24):
    """Get historical metrics"""
    return await monitoring_service.metrics_collector.get_historical_metrics(hours)

@app.get("/api/v1/alerts", response_model=List[PerformanceAlert])
async def get_active_alerts():
    """Get active performance alerts"""
    return await monitoring_service.metrics_collector.get_active_alerts()

@app.post("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve a performance alert"""
    success = await monitoring_service.metrics_collector.resolve_alert(alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert resolved successfully"}

@app.get("/api/v1/dashboard")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    return await monitoring_service.get_dashboard_data()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "monitoring_active": monitoring_service.collection_task is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)