"""
FraudGuard-360 Risk Scoring Service - Production Grade
=======================================================

Production-ready Risk Scoring Service with:
- Structured JSON logging
- Comprehensive error handling
- Real health checks
- Graceful shutdown

Author: FraudGuard-360 Platform Team
License: MIT
"""

import json
import signal
import sys
import time
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pydantic import BaseModel

# Add app directory to path for shared module access
sys.path.insert(0, os.path.dirname(__file__))

from shared.config import get_settings
from shared.logging_config import configure_logging

# Initialize settings
settings = get_settings()

# Configure structured logging
logger = configure_logging(
    service_name="risk-scoring-service",
    service_version=settings.service.version,
    log_level=settings.service.log_level,
    json_output=settings.service.environment != "development"
)

# Prometheus metrics
TRANSACTIONS_PROCESSED = Counter('risk_scoring_transactions_total', 'Transactions processed', ['status', 'risk_level'])
PROCESSING_DURATION = Histogram('risk_scoring_duration_seconds', 'Processing duration')
FRAUD_SCORE_DISTRIBUTION = Histogram('risk_scoring_fraud_score', 'Fraud score distribution')
ACTIVE_PROCESSING = Gauge('risk_scoring_active_processing', 'Active processing')
HIGH_RISK_TRANSACTIONS = Counter('risk_scoring_high_risk_total', 'High risk transactions')
DEPENDENCY_HEALTH = Gauge('risk_scoring_dependency_health', 'Dependency health', ['dependency'])
ERROR_COUNT = Counter('risk_scoring_errors_total', 'Errors by type', ['error_type'])


class DependencyHealth:
    """Track dependency health."""
    
    def __init__(self):
        self.kafka_consumer_healthy = False
        self.kafka_producer_healthy = False
        self.messages_processed = 0
        self.last_message_time = None
        self.last_check = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kafka_consumer": {"status": "healthy" if self.kafka_consumer_healthy else "unhealthy"},
            "kafka_producer": {"status": "healthy" if self.kafka_producer_healthy else "unhealthy"},
            "messages_processed": self.messages_processed,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "last_check": self.last_check.isoformat() if self.last_check else None
        }
    
    @property
    def is_healthy(self) -> bool:
        return self.kafka_consumer_healthy and self.kafka_producer_healthy


dependency_health = DependencyHealth()


class FraudDetectionEngine:
    """Production-grade fraud detection engine."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        logger.info("fraud_engine_initialized", rule_count=len(self.rules))
    
    def _initialize_rules(self) -> Dict[str, Any]:
        return {
            'high_amount_threshold': float(os.getenv('FRAUD_HIGH_AMOUNT', '5000.0')),
            'very_high_amount_threshold': float(os.getenv('FRAUD_VERY_HIGH_AMOUNT', '10000.0')),
            'suspicious_locations': os.getenv('FRAUD_SUSPICIOUS_LOCATIONS', 'Unknown,High Risk Zone').split(','),
            'suspicious_transaction_types': ['withdrawal', 'transfer'],
            'round_number_amounts': [1000, 2000, 5000, 10000]
        }
    
    def calculate_fraud_score(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate fraud score."""
        start_time = time.time()
        transaction_id = transaction.get('transaction_id', 'unknown')
        
        ACTIVE_PROCESSING.inc()
        
        try:
            fraud_score = 0.0
            risk_factors = []
            
            amount = float(transaction.get('amount', 0))
            location = transaction.get('location', 'Unknown')
            tx_type = transaction.get('transaction_type', 'purchase')
            
            # Rule 1: High amount
            if amount >= self.rules['very_high_amount_threshold']:
                fraud_score += 0.4
                risk_factors.append({'rule': 'very_high_amount', 'value': amount, 'score_impact': 0.4})
            elif amount >= self.rules['high_amount_threshold']:
                fraud_score += 0.2
                risk_factors.append({'rule': 'high_amount', 'value': amount, 'score_impact': 0.2})
            
            # Rule 2: Suspicious location
            if any(loc.lower() in location.lower() for loc in self.rules['suspicious_locations']):
                fraud_score += 0.25
                risk_factors.append({'rule': 'suspicious_location', 'value': location, 'score_impact': 0.25})
            
            # Rule 3: Suspicious transaction type
            if tx_type.lower() in self.rules['suspicious_transaction_types']:
                fraud_score += 0.15
                risk_factors.append({'rule': 'suspicious_transaction_type', 'value': tx_type, 'score_impact': 0.15})
            
            # Rule 4: Round number
            if amount in self.rules['round_number_amounts']:
                fraud_score += 0.1
                risk_factors.append({'rule': 'round_number', 'value': amount, 'score_impact': 0.1})
            
            fraud_score = min(fraud_score, 1.0)
            
            if fraud_score >= 0.7:
                risk_level, decision = "HIGH", "BLOCK"
                HIGH_RISK_TRANSACTIONS.inc()
            elif fraud_score >= 0.4:
                risk_level, decision = "MEDIUM", "REVIEW"
            else:
                risk_level, decision = "LOW", "APPROVE"
            
            processing_time = time.time() - start_time
            
            scored_transaction = {
                **transaction,
                'fraud_score': round(fraud_score, 4),
                'risk_level': risk_level,
                'decision': decision,
                'risk_factors': risk_factors,
                'scored_at': datetime.now(timezone.utc).isoformat(),
                'model_version': '1.0.0',
                'processing_time_ms': round(processing_time * 1000, 3)
            }
            
            FRAUD_SCORE_DISTRIBUTION.observe(fraud_score)
            PROCESSING_DURATION.observe(processing_time)
            TRANSACTIONS_PROCESSED.labels(status='success', risk_level=risk_level).inc()
            
            logger.info("transaction_scored", transaction_id=transaction_id, fraud_score=fraud_score,
                       risk_level=risk_level, decision=decision, processing_time_ms=round(processing_time * 1000, 3))
            
            return scored_transaction
            
        except Exception as e:
            logger.error("scoring_error", transaction_id=transaction_id, error=str(e), exc_info=e)
            TRANSACTIONS_PROCESSED.labels(status='error', risk_level='unknown').inc()
            ERROR_COUNT.labels(error_type='scoring').inc()
            raise
        finally:
            ACTIVE_PROCESSING.dec()


class RiskScoringService:
    """Main service class."""
    
    def __init__(self, setup_signals: bool = True):
        self.running = True
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.fraud_engine = FraudDetectionEngine()
        if setup_signals:
            self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        try:
            signal.signal(signal.SIGINT, self._shutdown_handler)
            signal.signal(signal.SIGTERM, self._shutdown_handler)
        except ValueError:
            # Signal handlers can only be set in main thread
            pass
    
    def _shutdown_handler(self, signum, frame):
        logger.info("shutdown_signal_received", signal=signal.Signals(signum).name)
        self.running = False
    
    def initialize_kafka(self):
        """Initialize Kafka with retry."""
        kafka_config = {'bootstrap_servers': settings.kafka.servers_list}
        
        if settings.kafka.security_protocol != "PLAINTEXT":
            kafka_config['security_protocol'] = settings.kafka.security_protocol
            if settings.kafka.sasl_mechanism:
                kafka_config['sasl_mechanism'] = settings.kafka.sasl_mechanism
                kafka_config['sasl_plain_username'] = settings.kafka.sasl_username
                kafka_config['sasl_plain_password'] = settings.kafka.sasl_password.get_secret_value()
        
        # Initialize consumer
        for attempt in range(5):
            try:
                self.consumer = KafkaConsumer(
                    settings.kafka.raw_transactions_topic,
                    **kafka_config,
                    group_id=settings.kafka.consumer_group,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset=settings.kafka.auto_offset_reset,
                    enable_auto_commit=True,
                    max_poll_records=settings.kafka.max_poll_records,
                    session_timeout_ms=settings.kafka.session_timeout_ms
                )
                dependency_health.kafka_consumer_healthy = True
                DEPENDENCY_HEALTH.labels(dependency="kafka_consumer").set(1)
                logger.info("kafka_consumer_connected", topic=settings.kafka.raw_transactions_topic)
                break
            except NoBrokersAvailable as e:
                logger.warning("kafka_consumer_retry", attempt=attempt + 1, error=str(e))
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))
                else:
                    dependency_health.kafka_consumer_healthy = False
                    DEPENDENCY_HEALTH.labels(dependency="kafka_consumer").set(0)
                    raise
        
        # Initialize producer
        for attempt in range(5):
            try:
                self.producer = KafkaProducer(
                    **kafka_config,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks=settings.kafka.acks,
                    retries=settings.kafka.retries,
                    max_in_flight_requests_per_connection=1
                )
                dependency_health.kafka_producer_healthy = True
                DEPENDENCY_HEALTH.labels(dependency="kafka_producer").set(1)
                logger.info("kafka_producer_connected", topic=settings.kafka.scored_transactions_topic)
                break
            except NoBrokersAvailable as e:
                logger.warning("kafka_producer_retry", attempt=attempt + 1, error=str(e))
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))
                else:
                    dependency_health.kafka_producer_healthy = False
                    DEPENDENCY_HEALTH.labels(dependency="kafka_producer").set(0)
                    raise
    
    def process_transaction(self, transaction: Dict[str, Any]):
        """Process a single transaction."""
        transaction_id = transaction.get('transaction_id', 'unknown')
        
        try:
            scored_transaction = self.fraud_engine.calculate_fraud_score(transaction)
            
            future = self.producer.send(settings.kafka.scored_transactions_topic, value=scored_transaction)
            record_metadata = future.get(timeout=10)
            
            dependency_health.messages_processed += 1
            dependency_health.last_message_time = datetime.now(timezone.utc)
            
            logger.info("scored_transaction_published", transaction_id=transaction_id,
                       topic=record_metadata.topic, partition=record_metadata.partition, offset=record_metadata.offset)
            
        except KafkaError as e:
            logger.error("kafka_publish_error", transaction_id=transaction_id, error=str(e), exc_info=e)
            ERROR_COUNT.labels(error_type='kafka_publish').inc()
            raise
        except Exception as e:
            logger.error("transaction_processing_error", transaction_id=transaction_id, error=str(e), exc_info=e)
            ERROR_COUNT.labels(error_type='processing').inc()
            raise
    
    def run(self):
        """Main service loop."""
        logger.info("service_starting", service="risk-scoring-service", environment=settings.service.environment)
        
        self.initialize_kafka()
        logger.info("service_ready", status="processing_transactions")
        
        try:
            while self.running:
                try:
                    messages = self.consumer.poll(timeout_ms=1000, max_records=10)
                    
                    for topic_partition, records in messages.items():
                        for record in records:
                            if not self.running:
                                break
                            try:
                                self.process_transaction(record.value)
                            except Exception as e:
                                logger.error("message_processing_error", error=str(e), exc_info=e)
                                continue
                except Exception as e:
                    if self.running:
                        logger.error("consumer_poll_error", error=str(e), exc_info=e)
                        time.sleep(1)
        except KeyboardInterrupt:
            logger.info("keyboard_interrupt_received")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("cleanup_starting")
        
        if self.consumer:
            try:
                self.consumer.close()
                logger.info("kafka_consumer_closed")
            except Exception as e:
                logger.error("consumer_close_error", error=str(e))
        
        if self.producer:
            try:
                self.producer.flush(timeout=5)
                self.producer.close(timeout=5)
                logger.info("kafka_producer_closed")
            except Exception as e:
                logger.error("producer_close_error", error=str(e))
        
        logger.info("cleanup_complete")
    
    def shutdown(self):
        """Signal the service to stop."""
        logger.info("shutdown_requested")
        self.running = False


# Global service instance for lifecycle management
_service_instance: Optional[RiskScoringService] = None
_consumer_thread: Optional[threading.Thread] = None


def run_kafka_consumer():
    """Run Kafka consumer in background thread."""
    global _service_instance
    try:
        # Don't setup signal handlers in thread
        _service_instance = RiskScoringService(setup_signals=False)
        _service_instance.run()
    except Exception as e:
        logger.error("kafka_consumer_failed", error=str(e))
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager."""
    global _consumer_thread, _service_instance
    
    logger.info("application_starting", service="risk-scoring-service")
    
    # Start Kafka consumer in background thread
    _consumer_thread = threading.Thread(target=run_kafka_consumer, daemon=True, name="kafka-consumer")
    _consumer_thread.start()
    logger.info("kafka_consumer_thread_started")
    
    # Wait briefly for Kafka initialization
    time.sleep(2)
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")
    if _service_instance:
        _service_instance.shutdown()
        # Give consumer thread time to cleanup
        if _consumer_thread and _consumer_thread.is_alive():
            _consumer_thread.join(timeout=5)
    logger.info("application_shutdown_complete")


# FastAPI application with lifespan
app = FastAPI(
    title="Risk Scoring Service",
    version=settings.service.version,
    lifespan=lifespan
)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str
    dependencies: Dict[str, Any]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    dependency_health.last_check = datetime.now(timezone.utc)
    
    if dependency_health.is_healthy:
        overall_status = "healthy"
    else:
        overall_status = "unhealthy"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                          detail=HealthResponse(
                              status=overall_status, service="risk-scoring-service",
                              version=settings.service.version,
                              timestamp=datetime.now(timezone.utc).isoformat(),
                              dependencies=dependency_health.to_dict()
                          ).model_dump())
    
    return HealthResponse(
        status=overall_status, service="risk-scoring-service", version=settings.service.version,
        timestamp=datetime.now(timezone.utc).isoformat(), dependencies=dependency_health.to_dict()
    )


@app.get("/ready")
async def readiness_check():
    if not dependency_health.is_healthy:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not ready")
    return {"status": "ready", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/live")
async def liveness_check():
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main():
    """Main entry point - run FastAPI with uvicorn."""
    import uvicorn
    logger.info("fraudguard_risk_scoring_service_starting")
    # Use app object directly to avoid double import
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


if __name__ == "__main__":
    main()
