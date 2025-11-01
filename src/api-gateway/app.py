"""
FraudGuard-360 API Gateway with Kafka Integration
==================================================

Production-ready API Gateway for fraud detection system.
Handles transaction ingestion and publishes to Kafka for processing.

Author: FraudGuard-360 Platform Team
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging
import json
import time
from typing import Optional
from datetime import datetime
import os

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
KAFKA_TOPIC = 'raw-transactions'

app = FastAPI(
    title="FraudGuard API Gateway",
    description="Real-time fraud detection API for telecom transactions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
transaction_count = Counter('transactions_total', 'Total transactions processed', ['status'])
request_duration = Histogram('request_duration_seconds', 'Request duration in seconds', ['endpoint'])

# Kafka producer (lazy initialization)
kafka_producer = None


def get_kafka_producer():
    """Get or create Kafka producer"""
    global kafka_producer
    if kafka_producer is None:
        try:
            kafka_producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"Kafka producer connected to {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            raise
    return kafka_producer


# Request/Response Models
class Transaction(BaseModel):
    """Transaction request model"""
    user_id: str = Field(..., description="Unique user identifier")
    amount: float = Field(..., ge=0, description="Transaction amount")
    location: str = Field(..., description="Transaction location")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    transaction_type: Optional[str] = Field("purchase", description="Type of transaction")
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address of transaction")

    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

    class Config:
        schema_extra = {
            "example": {
                "user_id": "USR_12345",
                "amount": 250.50,
                "location": "New York, NY",
                "merchant_id": "MERCH_789",
                "transaction_type": "purchase",
                "device_id": "DEV_456",
                "ip_address": "192.168.1.100"
            }
        }


class TransactionResponse(BaseModel):
    """Transaction response model"""
    transaction_id: str
    status: str
    message: str
    timestamp: str


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting API Gateway...")
    try:
        # Test Kafka connection
        get_kafka_producer()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.warning(f"Kafka not available during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global kafka_producer
    if kafka_producer:
        kafka_producer.close()
        logger.info("Kafka producer closed")


@app.get("/")
async def root():
    """Health check endpoint"""
    request_count.labels(method='GET', endpoint='/', status='200').inc()
    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "service": "api-gateway",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "kafka": "unknown"
        }
    }
    
    # Check Kafka connectivity
    try:
        producer = get_kafka_producer()
        health_status["checks"]["kafka"] = "healthy"
    except Exception as e:
        health_status["checks"]["kafka"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@app.post("/transactions", response_model=TransactionResponse, status_code=202)
async def create_transaction(transaction: Transaction, background_tasks: BackgroundTasks):
    """
    Submit a transaction for fraud analysis
    
    This endpoint accepts transaction data and publishes it to Kafka
    for asynchronous processing by the fraud detection pipeline.
    """
    start_time = time.time()
    transaction_id = f"TXN_{int(time.time() * 1000)}_{transaction.user_id}"
    
    try:
        # Enrich transaction data
        enriched_transaction = {
            "transaction_id": transaction_id,
            "user_id": transaction.user_id,
            "amount": transaction.amount,
            "location": transaction.location,
            "merchant_id": transaction.merchant_id,
            "transaction_type": transaction.transaction_type,
            "device_id": transaction.device_id,
            "ip_address": transaction.ip_address,
            "timestamp": datetime.utcnow().isoformat(),
            "currency": "USD",
            "status": "pending"
        }
        
        # Publish to Kafka
        producer = get_kafka_producer()
        future = producer.send(KAFKA_TOPIC, value=enriched_transaction)
        
        # Wait for confirmation (with timeout)
        record_metadata = future.get(timeout=10)
        
        logger.info(
            f"Transaction {transaction_id} published to Kafka topic {record_metadata.topic} "
            f"partition {record_metadata.partition} offset {record_metadata.offset}"
        )
        
        # Update metrics
        transaction_count.labels(status='success').inc()
        request_count.labels(method='POST', endpoint='/transactions', status='202').inc()
        
        duration = time.time() - start_time
        request_duration.labels(endpoint='/transactions').observe(duration)
        
        return TransactionResponse(
            transaction_id=transaction_id,
            status="accepted",
            message="Transaction submitted for fraud analysis",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except KafkaError as e:
        logger.error(f"Kafka error for transaction {transaction_id}: {e}")
        transaction_count.labels(status='kafka_error').inc()
        request_count.labels(method='POST', endpoint='/transactions', status='503').inc()
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable. Please try again later."
        )
    except Exception as e:
        logger.error(f"Error processing transaction {transaction_id}: {e}")
        transaction_count.labels(status='error').inc()
        request_count.labels(method='POST', endpoint='/transactions', status='500').inc()
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@app.get("/transactions/{transaction_id}")
async def get_transaction_status(transaction_id: str):
    """
    Get the status of a transaction
    
    Note: In production, this would query a database or cache.
    This is a placeholder implementation.
    """
    request_count.labels(method='GET', endpoint='/transactions/{id}', status='200').inc()
    
    return {
        "transaction_id": transaction_id,
        "status": "processing",
        "message": "Transaction is being analyzed",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
