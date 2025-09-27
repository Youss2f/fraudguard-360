from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import logging
import asyncio
import json
import random
import uuid
from datetime import datetime, timedelta
import os
from .monitoring.metrics import (
    get_metrics, monitor_api_request, update_health_metrics,
    record_fraud_detection, record_cdr_processing, record_database_operation
)
from .monitoring.health import (
    initialize_health_checker, get_system_health, get_component_health,
    HealthStatus, SystemHealth, ComponentHealth
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8001")
FLINK_URL = os.getenv("FLINK_URL", "http://flink-jobmanager:8081")
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://neo4j:7687")

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard 360 API Gateway",
    description="Central API gateway for fraud detection system",
    version="1.0.0"
)

# Initialize health checker and monitoring on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services and health monitoring on startup"""
    global health_checker
    logger.info("Starting FraudGuard API Gateway...")
    
    # Initialize health checker with default config
    config = {
        "health_check_interval": 30,
        "component_timeout": 5,
        "max_retries": 3
    }
    health_checker = initialize_health_checker(config)
    
    # Update initial health metrics
    update_health_metrics("api-gateway", True)
    
    # Start background health monitoring task
    asyncio.create_task(periodic_health_check())
    
    logger.info("FraudGuard API Gateway started with comprehensive monitoring")

async def periodic_health_check():
    """Background task for periodic health checks"""
    while True:
        try:
            # Check system health and update metrics
            update_health_metrics("system", True)
            await asyncio.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Health check task error: {e}")
            update_health_metrics("system", False)
            await asyncio.sleep(60)  # Retry after 1 minute on error

# CORS middleware - Enhanced with more origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://frontend:3000",
        "http://127.0.0.1:3000",
        "https://fraudguard-frontend.herokuapp.com",  # Production frontend
        "https://*.fraudguard360.com"  # Custom domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)

# Enhanced request middleware for tracking and security
@app.middleware("http")
async def add_request_middleware(request, call_next):
    """Add request tracking, timing, and security headers"""
    import time
    import uuid
    
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    
    # Start timing the request
    start_time = time.time()
    
    # Add request ID to headers
    request.state.request_id = request_id
    
    # Process the request
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise
    
    # Calculate response time
    process_time = time.time() - start_time
    
    # Add security and tracking headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{process_time:.4f}s"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Log request details
    logger.info(f"Request {request_id}: {request.method} {request.url.path} - {response.status_code} ({process_time:.4f}s)")
    
    return response

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed", response_model=SystemHealth)
async def detailed_health_check():
    """Detailed health check with all components"""
    try:
        health_status = await get_system_health()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

# Dashboard Endpoints
@app.get("/dashboard/kpis")
async def get_dashboard_kpis():
    """Get dashboard KPIs for real-time display"""
    import random
    import time
    
    return {
        "totalTransactions": random.randint(45000, 55000),
        "fraudAlerts": random.randint(150, 250),
        "riskScore": round(random.uniform(0.15, 0.35), 3),
        "successRate": round(random.uniform(0.92, 0.98), 3),
        "timestamp": time.time()
    }

@app.get("/dashboard/alerts")
async def get_dashboard_alerts():
    """Get recent fraud alerts"""
    import random
    from datetime import datetime, timedelta
    
    alert_types = ["High Risk Transaction", "Suspicious Pattern", "Account Takeover", "Card Testing", "Velocity Check Failed"]
    severities = ["critical", "high", "medium"]
    
    alerts = []
    for i in range(20):
        alert_time = datetime.utcnow() - timedelta(minutes=random.randint(1, 1440))
        alerts.append({
            "id": f"ALT-{random.randint(10000, 99999)}",
            "type": random.choice(alert_types),
            "severity": random.choice(severities),
            "amount": round(random.uniform(100, 50000), 2),
            "customer_id": f"CUST-{random.randint(1000, 9999)}",
            "timestamp": alert_time.isoformat(),
            "status": random.choice(["new", "investigating", "resolved"])
        })
    
    return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)

@app.get("/dashboard/transactions")
async def get_dashboard_transactions():
    """Get recent transactions for monitoring"""
    import random
    from datetime import datetime, timedelta
    
    transactions = []
    for i in range(50):
        trans_time = datetime.utcnow() - timedelta(minutes=random.randint(1, 60))
        transactions.append({
            "id": f"TXN-{random.randint(100000, 999999)}",
            "amount": round(random.uniform(10, 5000), 2),
            "customer_id": f"CUST-{random.randint(1000, 9999)}",
            "merchant": f"MERCHANT-{random.randint(100, 999)}",
            "risk_score": round(random.uniform(0.01, 0.95), 3),
            "status": random.choice(["approved", "declined", "pending"]),
            "timestamp": trans_time.isoformat()
        })
    
    return sorted(transactions, key=lambda x: x["timestamp"], reverse=True)

@app.get("/dashboard/charts")
async def get_dashboard_charts():
    """Get chart data for dashboard visualizations"""
    import random
    from datetime import datetime, timedelta
    
    # Generate hourly fraud data for the last 24 hours
    hourly_fraud = []
    for i in range(24):
        hour_time = datetime.utcnow() - timedelta(hours=23-i)
        hourly_fraud.append({
            "time": hour_time.strftime("%H:00"),
            "fraudCount": random.randint(5, 25),
            "totalTransactions": random.randint(1000, 3000)
        })
    
    # Generate transaction volume data
    transaction_volume = []
    for i in range(7):
        day_time = datetime.utcnow() - timedelta(days=6-i)
        transaction_volume.append({
            "day": day_time.strftime("%a"),
            "volume": random.randint(10000, 50000),
            "fraudulent": random.randint(100, 500)
        })
    
    # Generate risk distribution data
    risk_distribution = [
        {"range": "0-0.2", "count": random.randint(8000, 12000)},
        {"range": "0.2-0.4", "count": random.randint(3000, 5000)},
        {"range": "0.4-0.6", "count": random.randint(1000, 2000)},
        {"range": "0.6-0.8", "count": random.randint(500, 1000)},
        {"range": "0.8-1.0", "count": random.randint(100, 300)}
    ]
    
    return {
        "hourlyFraud": hourly_fraud,
        "transactionVolume": transaction_volume,
        "riskDistribution": risk_distribution
    }

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Authentication endpoints
@app.post("/auth/login")
async def login(credentials: dict):
    """Simple authentication endpoint"""
    username = credentials.get("username")
    password = credentials.get("password")
    
    # Simple demo credentials
    demo_users = {
        "admin": {"password": "admin123", "role": "administrator"},
        "analyst": {"password": "analyst123", "role": "analyst"},
        "viewer": {"password": "viewer123", "role": "viewer"}
    }
    
    if username in demo_users and demo_users[username]["password"] == password:
        import jwt
        import time
        
        # Create a simple JWT token (for demo purposes)
        token_data = {
            "username": username,
            "role": demo_users[username]["role"],
            "exp": time.time() + 3600  # 1 hour expiration
        }
        
        return {
            "success": True,
            "token": f"demo_token_{username}_{int(time.time())}",
            "user": {
                "username": username,
                "role": demo_users[username]["role"]
            }
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/auth/logout")
async def logout():
    """Logout endpoint"""
    return {"success": True, "message": "Logged out successfully"}

@app.get("/auth/verify")
async def verify_token(token: str = Query(...)):
    """Verify authentication token"""
    # Simple token verification for demo
    if token.startswith("demo_token_"):
        username = token.split("_")[2]
        return {
            "valid": True,
            "username": username,
            "role": "administrator" if username == "admin" else "fraud_analyst"
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            import random
            import time
            
            # Send KPI updates every 5 seconds
            kpi_update = {
                "type": "kpi_update",
                "data": {
                    "totalTransactions": random.randint(45000, 55000),
                    "fraudAlerts": random.randint(150, 250),
                    "riskScore": round(random.uniform(0.15, 0.35), 3),
                    "successRate": round(random.uniform(0.92, 0.98), 3),
                    "timestamp": time.time()
                }
            }
            await websocket.send_json(kpi_update)
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health/{component}", response_model=ComponentHealth)
async def component_health_check(component: str):
    """Health check for a specific component"""
    try:
        component_status = await get_component_health(component)
        if component_status is None:
            raise HTTPException(status_code=404, detail=f"Component {component} not found")
        return component_status
    except Exception as e:
        logger.error(f"Component health check failed: {e}")
        raise HTTPException(status_code=503, detail="Component health check failed")

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return await get_metrics()

# Data Models
class Alert(BaseModel):
    id: str
    case_id: str
    risk_score: float
    timestamp: datetime
    user_id: str
    fraud_type: str
    status: str
    assigned_analyst: Optional[str] = None
    priority: str
    description: str
    estimated_impact: float

class User(BaseModel):
    id: str
    name: str
    phone: str
    email: str
    account_created: datetime
    verification_status: str
    risk_level: str
    location: Dict[str, Any]

class Transaction(BaseModel):
    id: str
    user_id: str
    amount: float
    currency: str
    timestamp: datetime
    location: Dict[str, Any]
    suspicious: bool

class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    risk_score: float
    properties: Dict[str, Any]

class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relationship: str
    weight: float
    properties: Dict[str, Any]

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Generate mock data
def generate_mock_alerts(count: int = 50) -> List[Alert]:
    """Generate mock fraud alerts for demo purposes"""
    fraud_types = ["sim_box", "velocity_anomaly", "premium_rate", "location_anomaly", "device_cloning", "number_porting"]
    statuses = ["new", "assigned", "in_progress", "resolved", "false_positive"]
    priorities = ["critical", "high", "medium", "low"]
    analysts = ["Alice Johnson", "Bob Smith", "Carol Williams", "David Brown", None]
    
    alerts = []
    for i in range(count):
        alert = Alert(
            id=f"ALERT_{str(uuid.uuid4())[:8].upper()}",
            case_id=f"CASE_{str(uuid.uuid4())[:8].upper()}",
            risk_score=round(random.uniform(0.1, 1.0), 2),
            timestamp=datetime.utcnow() - timedelta(hours=random.randint(0, 72)),
            user_id=f"user_{random.randint(1000, 9999)}",
            fraud_type=random.choice(fraud_types),
            status=random.choice(statuses),
            assigned_analyst=random.choice(analysts),
            priority=random.choice(priorities),
            description=f"Suspicious activity detected: {random.choice(fraud_types).replace('_', ' ').title()}",
            estimated_impact=round(random.uniform(100, 50000), 2)
        )
        alerts.append(alert)
    
    return sorted(alerts, key=lambda x: x.risk_score, reverse=True)

def generate_mock_transactions(count: int = 100) -> List[Transaction]:
    """Generate mock transactions"""
    transactions = []
    currencies = ["USD", "EUR", "GBP", "CAD"]
    countries = ["US", "UK", "DE", "FR", "CA", "AU"]
    
    for i in range(count):
        transaction = Transaction(
            id=f"TXN_{str(uuid.uuid4())[:8].upper()}",
            user_id=f"user_{random.randint(1000, 9999)}",
            amount=round(random.uniform(10, 10000), 2),
            currency=random.choice(currencies),
            timestamp=datetime.utcnow() - timedelta(hours=random.randint(0, 168)),
            location={
                "country": random.choice(countries),
                "city": f"City_{random.randint(1, 100)}",
                "lat": round(random.uniform(-90, 90), 6),
                "lng": round(random.uniform(-180, 180), 6)
            },
            suspicious=random.choice([True, False])
        )
        transactions.append(transaction)
    
    return transactions

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send real-time updates every 5 seconds
            await asyncio.sleep(5)
            
            # Generate real-time data
            real_time_data = {
                "type": "real_time_update",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "transactions_per_second": round(random.uniform(10, 100), 1),
                    "active_alerts": random.randint(50, 200),
                    "detection_rate": round(random.uniform(85, 99), 1),
                    "financial_impact_saved": round(random.uniform(10000, 100000), 2),
                    "new_alerts": [
                        {
                            "id": f"ALERT_{str(uuid.uuid4())[:8].upper()}",
                            "risk_score": round(random.uniform(0.7, 1.0), 2),
                            "fraud_type": random.choice(["sim_box", "velocity_anomaly", "premium_rate"]),
                            "user_id": f"user_{random.randint(1000, 9999)}"
                        }
                        for _ in range(random.randint(0, 3))
                    ]
                }
            }
            
            await manager.broadcast(json.dumps(real_time_data))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/v1/alerts", response_model=List[Alert])
async def get_alerts(
    limit: int = Query(50, description="Number of alerts to return"),
    status: Optional[str] = Query(None, description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    analyst: Optional[str] = Query(None, description="Filter by assigned analyst")
):
    """Get fraud alerts with filtering"""
    try:
        alerts = generate_mock_alerts(limit)
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.status == status]
        if priority:
            alerts = [a for a in alerts if a.priority == priority]
        if analyst:
            alerts = [a for a in alerts if a.assigned_analyst == analyst]
            
        return alerts[:limit]
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {str(e)}")

@app.get("/api/v1/alerts/{alert_id}", response_model=Alert)
async def get_alert_details(alert_id: str):
    """Get detailed information about a specific alert"""
    try:
        # Generate mock alert details
        alert = Alert(
            id=alert_id,
            case_id=f"CASE_{str(uuid.uuid4())[:8].upper()}",
            risk_score=round(random.uniform(0.5, 1.0), 2),
            timestamp=datetime.utcnow() - timedelta(hours=random.randint(1, 24)),
            user_id=f"user_{random.randint(1000, 9999)}",
            fraud_type=random.choice(["sim_box", "velocity_anomaly", "premium_rate"]),
            status="new",
            assigned_analyst=None,
            priority="high",
            description="Detailed analysis of suspicious activity patterns detected",
            estimated_impact=round(random.uniform(1000, 25000), 2)
        )
        
        return alert
        
    except Exception as e:
        logger.error(f"Error getting alert details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alert details: {str(e)}")

@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_data():
    """Get aggregated data for the main dashboard"""
    try:
        dashboard_data = {
            "kpis": {
                "transactions_per_second": round(random.uniform(50, 150), 1),
                "active_alerts": random.randint(120, 180),
                "detection_rate": round(random.uniform(92, 98), 1),
                "financial_impact_saved": round(random.uniform(50000, 150000), 2),
                "cases_resolved_today": random.randint(15, 35),
                "users_at_risk": random.randint(45, 85)
            },
            "alerts_timeline": [
                {"hour": f"{i:02d}:00", "count": max(0, 15 + random.randint(-10, 25))} 
                for i in range(24)
            ],
            "fraud_distribution": [
                {"type": "SIM Box Fraud", "count": random.randint(25, 45), "percentage": 35.2},
                {"type": "Velocity Anomaly", "count": random.randint(20, 35), "percentage": 28.8},
                {"type": "Premium Rate", "count": random.randint(15, 25), "percentage": 18.5},
                {"type": "Location Anomaly", "count": random.randint(10, 20), "percentage": 12.1},
                {"type": "Device Cloning", "count": random.randint(5, 15), "percentage": 5.4}
            ],
            "geographic_data": [
                {"country": "United States", "lat": 39.8283, "lng": -98.5795, "alerts": random.randint(50, 100)},
                {"country": "United Kingdom", "lat": 55.3781, "lng": -3.4360, "alerts": random.randint(30, 60)},
                {"country": "Germany", "lat": 51.1657, "lng": 10.4515, "alerts": random.randint(25, 50)},
                {"country": "France", "lat": 46.2276, "lng": 2.2137, "alerts": random.randint(20, 40)},
                {"country": "Canada", "lat": 56.1304, "lng": -106.3468, "alerts": random.randint(15, 35)}
            ],
            "top_risk_users": [
                {
                    "user_id": f"user_{random.randint(1000, 9999)}", 
                    "risk_score": round(random.uniform(0.85, 0.99), 2), 
                    "alert_count": random.randint(5, 12),
                    "name": f"User {random.randint(1, 100)}",
                    "last_activity": (datetime.utcnow() - timedelta(minutes=random.randint(5, 120))).isoformat()
                }
                for _ in range(10)
            ]
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data retrieval failed: {str(e)}")

@app.get("/api/v1/graph/expand/{user_id}")
async def get_user_network_graph(user_id: str, depth: int = Query(2, description="Graph expansion depth")):
    """Get network graph data for a specific user"""
    try:
        # Generate mock graph data
        nodes = []
        edges = []
        
        # Central user node
        nodes.append(GraphNode(
            id=user_id,
            label=f"User {user_id[-4:]}",
            type="user",
            risk_score=round(random.uniform(0.7, 0.95), 2),
            properties={
                "phone": f"+1{random.randint(2000000000, 9999999999)}",
                "account_age": random.randint(30, 1000),
                "verification": random.choice(["verified", "pending", "failed"])
            }
        ))
        
        # Connected nodes
        for i in range(random.randint(5, 15)):
            node_id = f"node_{random.randint(1000, 9999)}"
            node_type = random.choice(["user", "device", "phone", "location"])
            
            nodes.append(GraphNode(
                id=node_id,
                label=f"{node_type.title()} {node_id[-4:]}",
                type=node_type,
                risk_score=round(random.uniform(0.1, 0.8), 2),
                properties={
                    "activity_count": random.randint(1, 100),
                    "first_seen": (datetime.utcnow() - timedelta(days=random.randint(1, 365))).isoformat()
                }
            ))
            
            # Create edge to central user
            edges.append(GraphEdge(
                id=f"edge_{user_id}_{node_id}",
                source=user_id,
                target=node_id,
                relationship=random.choice(["called", "shared_device", "same_location", "transaction"]),
                weight=round(random.uniform(0.1, 1.0), 2),
                properties={
                    "frequency": random.randint(1, 50),
                    "last_interaction": (datetime.utcnow() - timedelta(hours=random.randint(1, 72))).isoformat()
                }
            ))
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "center_node": user_id,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "depth": depth
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting network graph: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve network graph: {str(e)}")

@app.get("/api/v1/transactions", response_model=List[Transaction])
async def get_transactions(
    limit: int = Query(100, description="Number of transactions to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    suspicious_only: bool = Query(False, description="Return only suspicious transactions")
):
    """Get transaction data"""
    try:
        transactions = generate_mock_transactions(limit)
        
        if user_id:
            transactions = [t for t in transactions if t.user_id == user_id]
        if suspicious_only:
            transactions = [t for t in transactions if t.suspicious]
            
        return transactions[:limit]
        
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve transactions: {str(e)}")

@app.post("/api/v1/alerts/{alert_id}/assign")
async def assign_alert(alert_id: str, analyst: str):
    """Assign an alert to an analyst"""
    try:
        # In a real system, this would update the database
        logger.info(f"Alert {alert_id} assigned to {analyst}")
        return {"status": "success", "message": f"Alert {alert_id} assigned to {analyst}"}
    except Exception as e:
        logger.error(f"Error assigning alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign alert: {str(e)}")

@app.post("/api/v1/alerts/{alert_id}/status")
async def update_alert_status(alert_id: str, status: str):
    """Update alert status"""
    try:
        valid_statuses = ["new", "assigned", "in_progress", "resolved", "false_positive"]
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
            
        logger.info(f"Alert {alert_id} status updated to {status}")
        return {"status": "success", "message": f"Alert {alert_id} status updated to {status}"}
    except Exception as e:
        logger.error(f"Error updating alert status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update alert status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
