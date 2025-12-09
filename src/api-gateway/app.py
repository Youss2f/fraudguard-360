"""
FraudGuard-360 API Gateway - Telecom Fraud Detection (Zero Mock)
================================================================

Production-ready API Gateway for Telecom Fraud Detection with:
- Structured JSON logging
- Comprehensive error handling
- Real health checks for all dependencies
- Secure configuration management
- Full CRUD for CDRs (Call Detail Records)
- Analytics & statistics endpoints (Wangiri, SIM Box detection)
- Graph network data for visualization

Author: FraudGuard-360 Platform Team
License: MIT
"""

import asyncio
import time
import uuid
import sys
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, Response, status, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, desc, case, and_, or_, extract
from sqlalchemy.orm import Session

# Add app directory to path for shared module access
sys.path.insert(0, os.path.dirname(__file__))

from shared.config import get_settings
from shared.logging_config import configure_logging, LogContext
from shared.database import (
    get_session_factory, init_db, 
    CDR as CDRModel,
    Subscriber as SubscriberModel,
    CellTower as CellTowerModel,
    Alert as AlertModel,
    GraphNode, GraphEdge,
    CDRStatus, RiskLevel, AlertSeverity, CallType, FraudType
)

# Backward compatibility
TransactionModel = CDRModel
UserModel = SubscriberModel
MerchantModel = CellTowerModel
TransactionStatus = CDRStatus

# Initialize settings
settings = get_settings()

# Configure structured logging
logger = configure_logging(
    service_name="api-gateway",
    service_version=settings.service.version,
    log_level=settings.service.log_level,
    json_output=settings.service.environment != "development"
)

# Prometheus metrics
REQUEST_COUNT = Counter('api_gateway_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_gateway_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_CONNECTIONS = Gauge('api_gateway_active_connections', 'Active connections')
DEPENDENCY_HEALTH = Gauge('api_gateway_dependency_health', 'Dependency health', ['dependency'])
ERROR_COUNT = Counter('api_gateway_errors_total', 'Errors by type', ['error_type'])

# Global connections
kafka_producer: Optional[KafkaProducer] = None
redis_client: Optional[redis.Redis] = None
http_client: Optional[httpx.AsyncClient] = None
SessionLocal = None  # Database session factory


def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DependencyHealth:
    """Track dependency health status."""
    
    def __init__(self):
        self.kafka_healthy = False
        self.redis_healthy = False
        self.last_check = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kafka": {"status": "healthy" if self.kafka_healthy else "unhealthy"},
            "redis": {"status": "healthy" if self.redis_healthy else "unhealthy"},
            "last_check": self.last_check.isoformat() if self.last_check else None
        }
    
    @property
    def is_healthy(self) -> bool:
        return self.kafka_healthy and self.redis_healthy


dependency_health = DependencyHealth()


# Request/Response Models
class CDRRequest(BaseModel):
    """CDR request model for incoming call records."""
    
    caller_msisdn: str = Field(..., min_length=1, max_length=20)
    callee_msisdn: str = Field(..., min_length=1, max_length=20)
    duration: int = Field(..., ge=0, le=86400)
    cell_tower_id: Optional[str] = Field(None, max_length=50)
    call_type: str = Field(default="voice")
    roaming: bool = Field(default=False)
    imei: Optional[str] = Field(None, max_length=20)


# Backward compatibility alias
Transaction = CDRRequest


class CDRResponse(BaseModel):
    """CDR response model."""
    cdr_id: str
    status: str
    message: str
    timestamp: str


# Backward compatibility alias
TransactionResponse = CDRResponse


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    timestamp: str
    dependencies: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    request_id: Optional[str] = None
    timestamp: str


# ============================================================================
# PAGINATION & FILTER MODELS
# ============================================================================

class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class TransactionFilters(BaseModel):
    """CDR filtering options."""
    status: Optional[str] = None
    risk_level: Optional[str] = None
    fraud_type: Optional[str] = None
    call_type: Optional[str] = None
    caller_msisdn: Optional[str] = None
    callee_msisdn: Optional[str] = None
    cell_tower_id: Optional[str] = None
    imei: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    search: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class TransactionDetail(BaseModel):
    """Full CDR detail response."""
    id: str
    cdr_id: str
    caller_msisdn: str
    callee_msisdn: str
    duration: int
    call_type: str
    cell_tower_id: Optional[str]
    roaming: bool
    imei: Optional[str]
    fraud_type: str
    risk_score: float
    risk_level: str
    risk_factors: List[Dict[str, Any]]
    status: str
    decision: str
    reviewed_by: Optional[str]
    reviewed_at: Optional[str]
    review_notes: Optional[str]
    timestamp: str
    created_at: str
    processed_at: Optional[str]
    
    class Config:
        from_attributes = True


class TelecomStats(BaseModel):
    """Telecom statistics response."""
    total_calls: int
    total_sms: int
    simbox_detected: int
    wangiri_attempts: int
    avg_call_duration: float
    flagged_count: int
    approved_count: int
    declined_count: int
    pending_count: int
    under_review_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_risk_score: float
    flagged_rate: float
    calls_today: int
    calls_this_week: int
    revenue_at_risk: float  # Estimated revenue saved


# Backward compatibility alias
TransactionStats = TelecomStats


class TimeSeriesPoint(BaseModel):
    """Time series data point."""
    timestamp: str
    value: float
    label: Optional[str] = None


class RiskDistribution(BaseModel):
    """Risk distribution data."""
    low: int
    medium: int
    high: int
    critical: int


class DashboardMetrics(BaseModel):
    """Dashboard metrics response for Telecom."""
    total_calls: int
    total_volume_minutes: float
    avg_risk_score: float
    flagged_rate: float
    calls_trend: List[TimeSeriesPoint]
    volume_trend: List[TimeSeriesPoint]
    risk_distribution: RiskDistribution
    fraud_distribution: Dict[str, int]
    recent_alerts: List[Dict[str, Any]]
    top_risky_subscribers: List[Dict[str, Any]]
    top_cell_towers: List[Dict[str, Any]]


class GraphNetworkData(BaseModel):
    """Graph network response for visualization."""
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]


class ReviewRequest(BaseModel):
    """Transaction review request."""
    decision: str = Field(..., description="approve, decline, or flag")
    notes: Optional[str] = None
    reviewer: str


class AlertResponse(BaseModel):
    """Alert response."""
    id: str
    alert_id: str
    transaction_id: Optional[str]
    title: str
    description: Optional[str]
    severity: str
    category: Optional[str]
    is_read: bool
    is_resolved: bool
    created_at: str
    
    class Config:
        from_attributes = True


# Initialization functions
async def init_redis() -> Optional[redis.Redis]:
    """Initialize Redis with retry."""
    global redis_client
    
    for attempt in range(3):
        try:
            client = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                password=settings.redis.password.get_secret_value() if settings.redis.password else None,
                socket_timeout=settings.redis.socket_timeout,
                decode_responses=True
            )
            await client.ping()
            redis_client = client
            dependency_health.redis_healthy = True
            DEPENDENCY_HEALTH.labels(dependency="redis").set(1)
            logger.info("redis_connected", host=settings.redis.host)
            return client
        except Exception as e:
            logger.warning("redis_connection_failed", attempt=attempt + 1, error=str(e))
            if attempt < 2:
                await asyncio.sleep(1)
    
    dependency_health.redis_healthy = False
    DEPENDENCY_HEALTH.labels(dependency="redis").set(0)
    return None


def init_kafka() -> Optional[KafkaProducer]:
    """Initialize Kafka with retry."""
    global kafka_producer
    
    for attempt in range(3):
        try:
            producer = KafkaProducer(
                bootstrap_servers=settings.kafka.servers_list,
                value_serializer=lambda v: __import__('json').dumps(v).encode('utf-8'),
                acks=settings.kafka.acks,
                retries=settings.kafka.retries,
                max_in_flight_requests_per_connection=1
            )
            kafka_producer = producer
            dependency_health.kafka_healthy = True
            DEPENDENCY_HEALTH.labels(dependency="kafka").set(1)
            logger.info("kafka_connected", servers=settings.kafka.bootstrap_servers)
            return producer
        except NoBrokersAvailable as e:
            logger.warning("kafka_no_brokers", attempt=attempt + 1, error=str(e))
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            logger.error("kafka_connection_error", attempt=attempt + 1, error=str(e))
            if attempt < 2:
                time.sleep(2)
    
    dependency_health.kafka_healthy = False
    DEPENDENCY_HEALTH.labels(dependency="kafka").set(0)
    return None


async def init_http_client() -> httpx.AsyncClient:
    """Initialize HTTP client."""
    global http_client
    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout=30.0, connect=5.0, read=30.0, write=10.0, pool=5.0))
    http_client = client
    logger.info("http_client_initialized")
    return client


# Health checks
async def check_redis_health() -> bool:
    """Check Redis connectivity."""
    try:
        if redis_client:
            await redis_client.ping()
            dependency_health.redis_healthy = True
            DEPENDENCY_HEALTH.labels(dependency="redis").set(1)
            return True
    except Exception as e:
        logger.warning("redis_health_check_failed", error=str(e))
    
    dependency_health.redis_healthy = False
    DEPENDENCY_HEALTH.labels(dependency="redis").set(0)
    return False


def check_kafka_health() -> bool:
    """Check Kafka connectivity."""
    try:
        if kafka_producer and kafka_producer.bootstrap_connected():
            dependency_health.kafka_healthy = True
            DEPENDENCY_HEALTH.labels(dependency="kafka").set(1)
            return True
    except Exception as e:
        logger.warning("kafka_health_check_failed", error=str(e))
    
    dependency_health.kafka_healthy = False
    DEPENDENCY_HEALTH.labels(dependency="kafka").set(0)
    return False


async def run_health_checks():
    """Run all health checks."""
    await check_redis_health()
    check_kafka_health()
    dependency_health.last_check = datetime.now(timezone.utc)


async def periodic_health_checks():
    """Background health check task."""
    while True:
        try:
            await asyncio.sleep(30)
            await run_health_checks()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("periodic_health_check_error", error=str(e))


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global SessionLocal
    
    logger.info("application_starting", service="api-gateway", environment=settings.service.environment)
    
    # Initialize database
    try:
        init_db()
        SessionLocal = get_session_factory()
        logger.info("database_initialized")
    except Exception as e:
        logger.error("database_init_failed", error=str(e))
    
    # Initialize connections
    try:
        await init_redis()
    except Exception as e:
        logger.error("redis_init_failed", error=str(e))
    
    try:
        init_kafka()
    except Exception as e:
        logger.error("kafka_init_failed", error=str(e))
    
    await init_http_client()
    await run_health_checks()
    
    health_check_task = asyncio.create_task(periodic_health_checks())
    
    logger.info("application_started")
    
    yield
    
    logger.info("application_shutting_down")
    health_check_task.cancel()
    
    if kafka_producer:
        kafka_producer.close(timeout=5)
    if redis_client:
        await redis_client.close()
    if http_client:
        await http_client.aclose()
    
    logger.info("application_shutdown_complete")


# Create FastAPI application
app = FastAPI(
    title="FraudGuard-360 API Gateway",
    description="Production-grade API Gateway for fraud detection",
    version=settings.service.version,
    lifespan=lifespan,
    docs_url="/docs" if settings.service.debug else None,
    redoc_url="/redoc" if settings.service.debug else None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.service.debug else [],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", None)
    logger.warning("http_exception", status_code=exc.status_code, detail=exc.detail, request_id=request_id)
    ERROR_COUNT.labels(error_type=f"http_{exc.status_code}").inc()
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP_{exc.status_code}",
            message=exc.detail,
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.error("unhandled_exception", error_type=type(exc).__name__, error=str(exc), request_id=request_id)
    ERROR_COUNT.labels(error_type="unhandled").inc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )


# Request middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    correlation_id = request.headers.get("X-Correlation-ID", request_id)
    
    request.state.request_id = request_id
    request.state.correlation_id = correlation_id
    
    ACTIVE_CONNECTIONS.inc()
    start_time = time.time()
    
    with LogContext(request_id=request_id, correlation_id=correlation_id):
        logger.info("request_started", method=request.method, path=str(request.url.path))
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
            REQUEST_DURATION.labels(endpoint=request.url.path).observe(duration)
            
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            logger.info("request_completed", method=request.method, path=str(request.url.path), 
                       status_code=response.status_code, duration_ms=round(duration * 1000, 2))
            return response
        except Exception as e:
            logger.error("request_failed", error=str(e), exc_info=True)
            raise
        finally:
            ACTIVE_CONNECTIONS.dec()


# Endpoints
@app.get("/")
async def root():
    return {"status": "healthy", "service": "api-gateway", "version": settings.service.version,
            "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check - returns 503 if critical dependencies are down."""
    await run_health_checks()
    
    if dependency_health.is_healthy:
        overall_status = "healthy"
        status_code = status.HTTP_200_OK
    else:
        overall_status = "unhealthy"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    response = HealthResponse(
        status=overall_status,
        service="api-gateway",
        version=settings.service.version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dependencies=dependency_health.to_dict()
    )
    
    if status_code != status.HTTP_200_OK:
        logger.warning("health_check_failed", status=overall_status, dependencies=dependency_health.to_dict())
        raise HTTPException(status_code=status_code, detail=response.model_dump())
    
    return response


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    if not dependency_health.is_healthy:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready")
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/transactions", response_model=TransactionResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_transaction(transaction: Transaction, request: Request):
    """Submit a transaction for fraud analysis."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    transaction_id = f"TXN_{int(time.time() * 1000)}_{transaction.user_id}"
    
    logger.info("transaction_received", transaction_id=transaction_id, user_id=transaction.user_id,
                amount=transaction.amount, merchant_id=transaction.merchant_id)
    
    if not kafka_producer or not dependency_health.kafka_healthy:
        logger.error("kafka_unavailable", transaction_id=transaction_id)
        ERROR_COUNT.labels(error_type="kafka_unavailable").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                          detail="Message queue unavailable. Please retry later.")
    
    try:
        enriched_transaction = {
            "transaction_id": transaction_id,
            "request_id": request_id,
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "location": transaction.location,
            "merchant_id": transaction.merchant_id,
            "transaction_type": transaction.transaction_type,
            "device_id": transaction.device_id,
            "ip_address": transaction.ip_address or (request.client.host if request.client else None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "currency": "USD",
            "status": "pending"
        }
        
        future = kafka_producer.send(settings.kafka.raw_transactions_topic, value=enriched_transaction)
        record_metadata = future.get(timeout=10)
        
        logger.info("transaction_published", transaction_id=transaction_id, topic=record_metadata.topic,
                   partition=record_metadata.partition, offset=record_metadata.offset)
        
        return TransactionResponse(
            transaction_id=transaction_id,
            status="accepted",
            message="Transaction submitted for fraud analysis",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except KafkaError as e:
        logger.error("kafka_publish_failed", transaction_id=transaction_id, error=str(e), exc_info=True)
        ERROR_COUNT.labels(error_type="kafka_publish").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                          detail="Failed to queue transaction. Please retry later.")
    except Exception as e:
        logger.error("transaction_processing_error", transaction_id=transaction_id, error=str(e), exc_info=True)
        ERROR_COUNT.labels(error_type="transaction_processing").inc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process transaction")


@app.get("/transactions/{transaction_id}")
async def get_transaction_status(transaction_id: str):
    """Get transaction status."""
    if not redis_client or not dependency_health.redis_healthy:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cache service unavailable")
    
    try:
        cached_status = await redis_client.get(f"txn:{transaction_id}")
        if cached_status:
            return {"transaction_id": transaction_id, "status": cached_status,
                    "timestamp": datetime.now(timezone.utc).isoformat()}
        
        return {"transaction_id": transaction_id, "status": "processing",
                "message": "Transaction is being analyzed", "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error("transaction_status_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Failed to retrieve transaction status")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# V1 API ENDPOINTS - Zero Mock Implementation
# ============================================================================

@app.get("/v1/transactions", response_model=PaginatedResponse)
@app.get("/v1/cdrs", response_model=PaginatedResponse)
async def list_cdrs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("timestamp"),
    sort_order: SortOrder = Query(SortOrder.DESC),
    status: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    fraud_type: Optional[str] = Query(None),
    call_type: Optional[str] = Query(None),
    caller_msisdn: Optional[str] = Query(None),
    callee_msisdn: Optional[str] = Query(None),
    imei: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get paginated list of CDRs with filtering."""
    try:
        query = db.query(CDRModel)
        
        # Apply filters
        if status:
            try:
                status_enum = CDRStatus(status.lower())
                query = query.filter(CDRModel.status == status_enum)
            except ValueError:
                pass
        
        if risk_level:
            try:
                risk_enum = RiskLevel(risk_level.lower())
                query = query.filter(CDRModel.risk_level == risk_enum)
            except ValueError:
                pass
        
        if fraud_type:
            try:
                fraud_enum = FraudType(fraud_type.lower())
                query = query.filter(CDRModel.fraud_type == fraud_enum)
            except ValueError:
                pass
        
        if call_type:
            try:
                call_enum = CallType(call_type.lower())
                query = query.filter(CDRModel.call_type == call_enum)
            except ValueError:
                pass
        
        if caller_msisdn:
            query = query.filter(CDRModel.caller_msisdn == caller_msisdn)
        
        if callee_msisdn:
            query = query.filter(CDRModel.callee_msisdn == callee_msisdn)
        
        if imei:
            query = query.filter(CDRModel.imei == imei)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    CDRModel.cdr_id.ilike(search_term),
                    CDRModel.caller_msisdn.ilike(search_term),
                    CDRModel.callee_msisdn.ilike(search_term),
                    CDRModel.imei.ilike(search_term),
                    CDRModel.cell_tower_id.ilike(search_term),
                )
            )
        
        # Get total count
        total = query.count()
        
        # Apply sorting
        sort_column = getattr(CDRModel, sort_by, CDRModel.timestamp)
        if sort_order == SortOrder.DESC:
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(sort_column)
        
        # Apply pagination
        offset = (page - 1) * page_size
        cdrs = query.offset(offset).limit(page_size).all()
        
        # Convert to response format
        data = []
        for cdr in cdrs:
            data.append({
                "id": str(cdr.id),
                "cdrId": cdr.cdr_id,
                "callerMsisdn": cdr.caller_msisdn,
                "calleeMsisdn": cdr.callee_msisdn,
                "duration": cdr.duration,
                "callType": cdr.call_type.value if cdr.call_type else "voice",
                "cellTowerId": cdr.cell_tower_id,
                "roaming": cdr.roaming,
                "imei": cdr.imei,
                "fraudType": cdr.fraud_type.value if cdr.fraud_type else "none",
                "riskScore": cdr.risk_score,
                "riskLevel": cdr.risk_level.value if cdr.risk_level else "low",
                "riskFactors": cdr.risk_factors or [],
                "status": cdr.status.value if cdr.status else "pending",
                "decision": cdr.decision,
                "reviewedBy": cdr.reviewed_by,
                "reviewedAt": cdr.reviewed_at.isoformat() if cdr.reviewed_at else None,
                "reviewNotes": cdr.review_notes,
                "timestamp": cdr.timestamp.isoformat() if cdr.timestamp else None,
                "createdAt": cdr.created_at.isoformat() if cdr.created_at else None,
                "processedAt": cdr.processed_at.isoformat() if cdr.processed_at else None,
            })
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except Exception as e:
        logger.error("list_cdrs_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch CDRs")


@app.get("/v1/transactions/{transaction_id}")
@app.get("/v1/cdrs/{cdr_id}")
async def get_cdr_detail(transaction_id: str = None, cdr_id: str = None, db: Session = Depends(get_db)):
    """Get detailed CDR by ID."""
    try:
        record_id = cdr_id or transaction_id
        cdr = db.query(CDRModel).filter(
            CDRModel.cdr_id == record_id
        ).first()
        
        if not cdr:
            raise HTTPException(status_code=404, detail="CDR not found")
        
        return {
            "id": str(cdr.id),
            "cdrId": cdr.cdr_id,
            "callerMsisdn": cdr.caller_msisdn,
            "calleeMsisdn": cdr.callee_msisdn,
            "duration": cdr.duration,
            "callType": cdr.call_type.value if cdr.call_type else "voice",
            "cellTowerId": cdr.cell_tower_id,
            "roaming": cdr.roaming,
            "imei": cdr.imei,
            "fraudType": cdr.fraud_type.value if cdr.fraud_type else "none",
            "riskScore": cdr.risk_score,
            "riskLevel": cdr.risk_level.value if cdr.risk_level else "low",
            "riskFactors": cdr.risk_factors or [],
            "status": cdr.status.value if cdr.status else "pending",
            "decision": cdr.decision,
            "reviewedBy": cdr.reviewed_by,
            "reviewedAt": cdr.reviewed_at.isoformat() if cdr.reviewed_at else None,
            "reviewNotes": cdr.review_notes,
            "timestamp": cdr.timestamp.isoformat() if cdr.timestamp else None,
            "createdAt": cdr.created_at.isoformat() if cdr.created_at else None,
            "processedAt": cdr.processed_at.isoformat() if cdr.processed_at else None,
            "modelVersion": cdr.model_version,
            "processingTimeMs": cdr.processing_time_ms,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_cdr_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch CDR")


@app.post("/v1/transactions/{transaction_id}/review")
@app.post("/v1/cdrs/{cdr_id}/review")
async def review_cdr(
    transaction_id: str = None,
    cdr_id: str = None,
    review: ReviewRequest = None,
    db: Session = Depends(get_db)
):
    """Review and update CDR status."""
    try:
        record_id = cdr_id or transaction_id
        cdr = db.query(CDRModel).filter(
            CDRModel.cdr_id == record_id
        ).first()
        
        if not cdr:
            raise HTTPException(status_code=404, detail="CDR not found")
        
        # Update status based on decision
        decision_map = {
            "approve": (CDRStatus.APPROVED, "approve"),
            "decline": (CDRStatus.DECLINED, "block"),
            "flag": (CDRStatus.FLAGGED, "block"),
        }
        
        if review.decision.lower() not in decision_map:
            raise HTTPException(status_code=400, detail="Invalid decision. Use: approve, decline, or flag")
        
        new_status, new_decision = decision_map[review.decision.lower()]
        
        cdr.status = new_status
        cdr.decision = new_decision
        cdr.reviewed_by = review.reviewer
        cdr.reviewed_at = datetime.now(timezone.utc)
        cdr.review_notes = review.notes
        
        db.commit()
        
        logger.info("cdr_reviewed", 
                   cdr_id=record_id, 
                   decision=review.decision,
                   reviewer=review.reviewer)
        
        return {
            "success": True,
            "cdr_id": record_id,
            "new_status": new_status.value,
            "reviewed_by": review.reviewer,
            "reviewed_at": cdr.reviewed_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("review_cdr_error", error=str(e), exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to review CDR")


@app.get("/v1/analytics/stats", response_model=TelecomStats)
async def get_telecom_stats(db: Session = Depends(get_db)):
    """Get comprehensive telecom statistics."""
    try:
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=now.weekday())
        
        # Basic counts
        total = db.query(func.count(CDRModel.id)).scalar() or 0
        
        # Call type counts
        total_voice = db.query(func.count(CDRModel.id)).filter(
            CDRModel.call_type == CallType.VOICE
        ).scalar() or 0
        
        total_sms = db.query(func.count(CDRModel.id)).filter(
            CDRModel.call_type == CallType.SMS
        ).scalar() or 0
        
        # Total duration and average
        total_duration = db.query(func.sum(CDRModel.duration)).filter(
            CDRModel.call_type == CallType.VOICE
        ).scalar() or 0
        avg_duration = db.query(func.avg(CDRModel.duration)).filter(
            CDRModel.call_type == CallType.VOICE, CDRModel.duration > 0
        ).scalar() or 0.0
        avg_risk = db.query(func.avg(CDRModel.risk_score)).scalar() or 0.0
        
        # Fraud type counts
        simbox_detected = db.query(func.count(CDRModel.id)).filter(
            CDRModel.fraud_type == FraudType.SIMBOX
        ).scalar() or 0
        
        wangiri_attempts = db.query(func.count(CDRModel.id)).filter(
            CDRModel.fraud_type == FraudType.WANGIRI
        ).scalar() or 0
        
        # Status counts
        flagged = db.query(func.count(CDRModel.id)).filter(
            CDRModel.status == CDRStatus.FLAGGED
        ).scalar() or 0
        
        approved = db.query(func.count(CDRModel.id)).filter(
            CDRModel.status == CDRStatus.APPROVED
        ).scalar() or 0
        
        declined = db.query(func.count(CDRModel.id)).filter(
            CDRModel.status == CDRStatus.DECLINED
        ).scalar() or 0
        
        pending = db.query(func.count(CDRModel.id)).filter(
            CDRModel.status == CDRStatus.PENDING
        ).scalar() or 0
        
        under_review = db.query(func.count(CDRModel.id)).filter(
            CDRModel.status == CDRStatus.UNDER_REVIEW
        ).scalar() or 0
        
        # Risk level counts
        high_risk = db.query(func.count(CDRModel.id)).filter(
            CDRModel.risk_level == RiskLevel.HIGH
        ).scalar() or 0
        
        medium_risk = db.query(func.count(CDRModel.id)).filter(
            CDRModel.risk_level == RiskLevel.MEDIUM
        ).scalar() or 0
        
        low_risk = db.query(func.count(CDRModel.id)).filter(
            CDRModel.risk_level == RiskLevel.LOW
        ).scalar() or 0
        
        # Time-based stats
        calls_today = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= today_start
        ).scalar() or 0
        
        calls_week = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= week_start
        ).scalar() or 0
        
        flagged_rate = (flagged + under_review) / total * 100 if total > 0 else 0.0
        
        # Estimate revenue saved (blocked fraud calls * avg rate)
        revenue_at_risk = (simbox_detected + wangiri_attempts) * 0.50  # $0.50 per fraudulent call
        
        return TelecomStats(
            total_calls=total_voice,
            total_sms=total_sms,
            simbox_detected=simbox_detected,
            wangiri_attempts=wangiri_attempts,
            avg_call_duration=round(avg_duration, 2),
            flagged_count=flagged,
            approved_count=approved,
            declined_count=declined,
            pending_count=pending,
            under_review_count=under_review,
            high_risk_count=high_risk,
            medium_risk_count=medium_risk,
            low_risk_count=low_risk,
            avg_risk_score=round(avg_risk, 4),
            flagged_rate=round(flagged_rate, 2),
            calls_today=calls_today,
            calls_this_week=calls_week,
            revenue_at_risk=round(revenue_at_risk, 2)
        )
        
    except Exception as e:
        logger.error("get_stats_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


@app.get("/v1/analytics/dashboard", response_model=DashboardMetrics)
async def get_dashboard_metrics(
    days: int = Query(30, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """Get dashboard metrics with trends for Telecom."""
    try:
        now = datetime.now(timezone.utc)
        start_date = now - timedelta(days=days)
        
        # Basic metrics
        total = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date
        ).scalar() or 0
        
        total_duration = db.query(func.sum(CDRModel.duration)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.call_type == CallType.VOICE
        ).scalar() or 0
        total_volume_minutes = total_duration / 60.0
        
        avg_risk = db.query(func.avg(CDRModel.risk_score)).filter(
            CDRModel.created_at >= start_date
        ).scalar() or 0.0
        
        flagged = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.status.in_([CDRStatus.FLAGGED, CDRStatus.UNDER_REVIEW])
        ).scalar() or 0
        
        flagged_rate = flagged / total * 100 if total > 0 else 0.0
        
        # Daily call trends (last 7 days)
        calls_trend = []
        volume_trend = []
        for i in range(7):
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            day_count = db.query(func.count(CDRModel.id)).filter(
                CDRModel.created_at >= day_start,
                CDRModel.created_at < day_end
            ).scalar() or 0
            
            day_duration = db.query(func.sum(CDRModel.duration)).filter(
                CDRModel.created_at >= day_start,
                CDRModel.created_at < day_end,
                CDRModel.call_type == CallType.VOICE
            ).scalar() or 0
            
            calls_trend.append(TimeSeriesPoint(
                timestamp=day_start.isoformat(),
                value=float(day_count),
                label=day_start.strftime("%a")
            ))
            
            volume_trend.append(TimeSeriesPoint(
                timestamp=day_start.isoformat(),
                value=round(day_duration / 60.0, 2),  # Minutes
                label=day_start.strftime("%a")
            ))
        
        calls_trend.reverse()
        volume_trend.reverse()
        
        # Risk distribution
        low = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.risk_level == RiskLevel.LOW
        ).scalar() or 0
        
        medium = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.risk_level == RiskLevel.MEDIUM
        ).scalar() or 0
        
        high = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.risk_level == RiskLevel.HIGH
        ).scalar() or 0
        
        critical = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.risk_level == RiskLevel.CRITICAL
        ).scalar() or 0
        
        # Fraud type distribution
        wangiri = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.fraud_type == FraudType.WANGIRI
        ).scalar() or 0
        
        simbox = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.fraud_type == FraudType.SIMBOX
        ).scalar() or 0
        
        irsf = db.query(func.count(CDRModel.id)).filter(
            CDRModel.created_at >= start_date,
            CDRModel.fraud_type == FraudType.IRSF
        ).scalar() or 0
        
        fraud_distribution = {
            "wangiri": wangiri,
            "simbox": simbox,
            "irsf": irsf,
            "normal": total - wangiri - simbox - irsf
        }
        
        # Recent alerts
        recent_alerts_query = db.query(AlertModel).order_by(
            desc(AlertModel.created_at)
        ).limit(10).all()
        
        recent_alerts = [
            {
                "id": str(alert.id),
                "alertId": alert.alert_id,
                "title": alert.title,
                "severity": alert.severity.value if alert.severity else "info",
                "category": alert.category,
                "isRead": alert.is_read,
                "createdAt": alert.created_at.isoformat() if alert.created_at else None
            }
            for alert in recent_alerts_query
        ]
        
        # Top risky subscribers
        risky_subscribers_query = db.query(SubscriberModel).filter(
            SubscriberModel.risk_score > 0
        ).order_by(desc(SubscriberModel.risk_score)).limit(5).all()
        
        top_risky_subscribers = [
            {
                "msisdn": sub.msisdn,
                "name": sub.name,
                "riskScore": sub.risk_score,
                "totalCalls": sub.total_calls,
                "flaggedCalls": sub.flagged_calls
            }
            for sub in risky_subscribers_query
        ]
        
        # Top cell towers by activity
        top_towers_query = db.query(CellTowerModel).order_by(
            desc(CellTowerModel.total_calls)
        ).limit(5).all()
        
        top_cell_towers = [
            {
                "towerId": tower.tower_id,
                "name": tower.name,
                "location": tower.location,
                "totalCalls": tower.total_calls,
                "riskScore": tower.risk_score
            }
            for tower in top_towers_query
        ]
        
        return DashboardMetrics(
            total_calls=total,
            total_volume_minutes=round(total_volume_minutes, 2),
            avg_risk_score=round(avg_risk, 4),
            flagged_rate=round(flagged_rate, 2),
            calls_trend=calls_trend,
            volume_trend=volume_trend,
            risk_distribution=RiskDistribution(
                low=low,
                medium=medium,
                high=high,
                critical=critical
            ),
            fraud_distribution=fraud_distribution,
            recent_alerts=recent_alerts,
            top_risky_subscribers=top_risky_subscribers,
            top_cell_towers=top_cell_towers
        )
        
    except Exception as e:
        logger.error("get_dashboard_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch dashboard metrics")


@app.get("/v1/graph/network", response_model=GraphNetworkData)
async def get_graph_network(
    node_type: Optional[str] = Query(None, description="Filter by node type: user, merchant, device, ip"),
    limit: int = Query(200, ge=10, le=500),
    min_risk: float = Query(0.0, ge=0, le=1),
    db: Session = Depends(get_db)
):
    """Get graph network data for visualization."""
    try:
        # Query nodes
        nodes_query = db.query(GraphNode)
        
        if node_type:
            nodes_query = nodes_query.filter(GraphNode.node_type == node_type)
        
        if min_risk > 0:
            nodes_query = nodes_query.filter(GraphNode.risk_score >= min_risk)
        
        nodes_result = nodes_query.limit(limit).all()
        
        # Get node IDs for edge filtering
        node_ids = {node.node_id for node in nodes_result}
        
        # Query edges connected to these nodes
        edges_query = db.query(GraphEdge).filter(
            or_(
                GraphEdge.source_id.in_(node_ids),
                GraphEdge.target_id.in_(node_ids)
            )
        ).limit(limit * 3)  # More edges than nodes typically
        
        edges_result = edges_query.all()
        
        # Build response
        nodes = []
        for node in nodes_result:
            nodes.append({
                "id": node.node_id,
                "type": node.node_type,
                "label": node.label or node.node_id,
                "riskScore": node.risk_score,
                "properties": node.properties or {},
                # Visual properties based on type
                "color": _get_node_color(node.node_type, node.risk_score),
                "size": _get_node_size(node.node_type),
            })
        
        links = []
        seen_edges = set()
        for edge in edges_result:
            # Only include edges where both nodes are in our result
            if edge.source_id in node_ids and edge.target_id in node_ids:
                edge_key = f"{edge.source_id}-{edge.target_id}-{edge.edge_type}"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    links.append({
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "type": edge.edge_type,
                        "weight": edge.weight,
                        "properties": edge.properties or {},
                    })
        
        return GraphNetworkData(nodes=nodes, links=links)
        
    except Exception as e:
        logger.error("get_graph_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch graph data")


def _get_node_color(node_type: str, risk_score: float) -> str:
    """Get node color based on type and risk."""
    if risk_score >= 0.7:
        return "#ef4444"  # Red for high risk
    elif risk_score >= 0.4:
        return "#f59e0b"  # Orange for medium risk
    
    type_colors = {
        "msisdn": "#3b82f6",      # Blue for phone numbers
        "cell_tower": "#10b981",  # Green for towers
        "imei": "#8b5cf6",        # Purple for devices
        "subscriber": "#3b82f6",   # Blue for subscribers
    }
    return type_colors.get(node_type, "#6b7280")


def _get_node_size(node_type: str) -> int:
    """Get node size based on type."""
    size_map = {
        "msisdn": 8,
        "cell_tower": 10,
        "imei": 6,
        "subscriber": 8,
    }
    return size_map.get(node_type, 6)


@app.get("/v1/alerts", response_model=PaginatedResponse)
async def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    severity: Optional[str] = Query(None),
    is_read: Optional[bool] = Query(None),
    is_resolved: Optional[bool] = Query(None),
    db: Session = Depends(get_db)
):
    """Get paginated list of alerts."""
    try:
        query = db.query(AlertModel)
        
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
                query = query.filter(AlertModel.severity == severity_enum)
            except ValueError:
                pass
        
        if is_read is not None:
            query = query.filter(AlertModel.is_read == is_read)
        
        if is_resolved is not None:
            query = query.filter(AlertModel.is_resolved == is_resolved)
        
        total = query.count()
        
        offset = (page - 1) * page_size
        alerts = query.order_by(desc(AlertModel.created_at)).offset(offset).limit(page_size).all()
        
        data = [
            {
                "id": str(alert.id),
                "alertId": alert.alert_id,
                "transactionId": alert.transaction_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value if alert.severity else "info",
                "category": alert.category,
                "isRead": alert.is_read,
                "isResolved": alert.is_resolved,
                "resolvedBy": alert.resolved_by,
                "resolvedAt": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "createdAt": alert.created_at.isoformat() if alert.created_at else None,
            }
            for alert in alerts
        ]
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except Exception as e:
        logger.error("list_alerts_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch alerts")


@app.post("/v1/alerts/{alert_id}/read")
async def mark_alert_read(alert_id: str, db: Session = Depends(get_db)):
    """Mark alert as read."""
    try:
        alert = db.query(AlertModel).filter(AlertModel.alert_id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.is_read = True
        db.commit()
        
        return {"success": True, "alert_id": alert_id, "is_read": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("mark_alert_read_error", error=str(e), exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update alert")


@app.get("/v1/users", response_model=PaginatedResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_risk: Optional[float] = Query(None),
    account_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get paginated list of users."""
    try:
        query = db.query(UserModel)
        
        if min_risk is not None:
            query = query.filter(UserModel.risk_score >= min_risk)
        
        if account_type:
            query = query.filter(UserModel.account_type == account_type)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    UserModel.user_id.ilike(search_term),
                    UserModel.name.ilike(search_term),
                    UserModel.email.ilike(search_term),
                )
            )
        
        total = query.count()
        
        offset = (page - 1) * page_size
        users = query.order_by(desc(UserModel.risk_score)).offset(offset).limit(page_size).all()
        
        data = [
            {
                "id": str(user.id),
                "userId": user.user_id,
                "email": user.email,
                "name": user.name,
                "accountType": user.account_type,
                "riskScore": user.risk_score,
                "totalTransactions": user.total_transactions,
                "flaggedTransactions": user.flagged_transactions,
                "createdAt": user.created_at.isoformat() if user.created_at else None,
            }
            for user in users
        ]
        
        total_pages = (total + page_size - 1) // page_size
        
        return PaginatedResponse(
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )
        
    except Exception as e:
        logger.error("list_users_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch users")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.service.host, port=settings.service.port,
                workers=settings.service.workers, log_level=settings.service.log_level.lower(),
                reload=settings.service.debug)
