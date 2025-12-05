"""
FraudGuard-360 API Gateway - Production Grade
==============================================

Production-ready API Gateway with:
- Structured JSON logging
- Comprehensive error handling
- Real health checks for all dependencies
- Secure configuration management

Author: FraudGuard-360 Platform Team
License: MIT
"""

import asyncio
import time
import uuid
import sys
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field, field_validator

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.config import get_settings
from shared.logging_config import configure_logging, LogContext

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
class Transaction(BaseModel):
    """Transaction request model."""
    
    user_id: str = Field(..., min_length=1, max_length=100)
    amount: float = Field(..., ge=0, le=1000000)
    location: str = Field(..., min_length=1, max_length=200)
    merchant_id: str = Field(..., min_length=1, max_length=100)
    transaction_type: str = Field(default="purchase")
    device_id: Optional[str] = Field(None, max_length=100)
    ip_address: Optional[str] = Field(None)
    
    @field_validator('amount')
    @classmethod
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return round(v, 2)


class TransactionResponse(BaseModel):
    """Transaction response model."""
    transaction_id: str
    status: str
    message: str
    timestamp: str


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
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0))
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
    logger.info("application_starting", service="api-gateway", environment=settings.service.environment)
    
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
    logger.error("unhandled_exception", error_type=type(exc).__name__, error=str(exc), request_id=request_id, exc_info=exc)
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
            logger.error("request_failed", error=str(e), exc_info=e)
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
        logger.error("kafka_publish_failed", transaction_id=transaction_id, error=str(e), exc_info=e)
        ERROR_COUNT.labels(error_type="kafka_publish").inc()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                          detail="Failed to queue transaction. Please retry later.")
    except Exception as e:
        logger.error("transaction_processing_error", transaction_id=transaction_id, error=str(e), exc_info=e)
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
        logger.error("transaction_status_error", error=str(e), exc_info=e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                          detail="Failed to retrieve transaction status")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.service.host, port=settings.service.port,
                workers=settings.service.workers, log_level=settings.service.log_level.lower(),
                reload=settings.service.debug)
