"""
FraudGuard-360 Risk Scoring Service
====================================

Consumes transactions from Kafka, applies rule-based fraud detection,
and publishes scored transactions back to Kafka.

Author: FraudGuard-360 Platform Team
"""

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import json
import logging
import time
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
INPUT_TOPIC = 'raw-transactions'
OUTPUT_TOPIC = 'scored-transactions'
CONSUMER_GROUP = 'risk-scoring-group'
METRICS_PORT = 8001

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
transactions_processed = Counter('transactions_processed_total', 'Total transactions processed', ['status'])
fraud_score_histogram = Histogram('fraud_score', 'Distribution of fraud scores')
processing_time = Histogram('transaction_processing_seconds', 'Transaction processing time')
high_risk_transactions = Counter('high_risk_transactions_total', 'Total high-risk transactions detected')
active_transactions = Gauge('active_transactions', 'Currently processing transactions')


class FraudDetectionEngine:
    """Simple rule-based fraud detection engine"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        logger.info("Fraud detection engine initialized")
    
    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize fraud detection rules"""
        return {
            'high_amount_threshold': 1000.0,
            'very_high_amount_threshold': 5000.0,
            'suspicious_locations': ['Unknown', 'High Risk Zone'],
            'velocity_check_enabled': True,
            'amount_deviation_factor': 3.0
        }
    
    def calculate_fraud_score(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate fraud score based on transaction attributes
        
        Returns enriched transaction with fraud_score and risk_level
        """
        start_time = time.time()
        
        try:
            amount = float(transaction.get('amount', 0))
            location = transaction.get('location', 'Unknown')
            transaction_type = transaction.get('transaction_type', 'unknown')
            
            # Initialize score
            fraud_score = 0.0
            risk_factors = []
            
            # Rule 1: High Amount Check
            if amount > self.rules['very_high_amount_threshold']:
                fraud_score += 0.5
                risk_factors.append(f"Very high amount: ${amount}")
            elif amount > self.rules['high_amount_threshold']:
                fraud_score += 0.3
                risk_factors.append(f"High amount: ${amount}")
            
            # Rule 2: Suspicious Location Check
            if location in self.rules['suspicious_locations']:
                fraud_score += 0.2
                risk_factors.append(f"Suspicious location: {location}")
            
            # Rule 3: Transaction Type Risk
            high_risk_types = ['withdrawal', 'transfer', 'cash_advance']
            if transaction_type in high_risk_types:
                fraud_score += 0.15
                risk_factors.append(f"High-risk transaction type: {transaction_type}")
            
            # Rule 4: Round Number Check (common in fraud)
            if amount % 100 == 0 and amount >= 500:
                fraud_score += 0.1
                risk_factors.append("Round number transaction")
            
            # Normalize score to 0-1 range
            fraud_score = min(fraud_score, 1.0)
            
            # Determine risk level
            if fraud_score >= 0.7:
                risk_level = "HIGH"
                decision = "BLOCK"
                high_risk_transactions.inc()
            elif fraud_score >= 0.4:
                risk_level = "MEDIUM"
                decision = "REVIEW"
            else:
                risk_level = "LOW"
                decision = "APPROVE"
            
            # Enrich transaction
            scored_transaction = transaction.copy()
            scored_transaction.update({
                'fraud_score': round(fraud_score, 3),
                'risk_level': risk_level,
                'decision': decision,
                'risk_factors': risk_factors,
                'scored_at': datetime.utcnow().isoformat(),
                'model_version': '1.0.0',
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            })
            
            # Update metrics
            fraud_score_histogram.observe(fraud_score)
            processing_time.observe(time.time() - start_time)
            
            logger.info(
                f"Transaction {transaction.get('transaction_id')} scored: "
                f"{fraud_score:.3f} ({risk_level}) - Decision: {decision}"
            )
            
            return scored_transaction
            
        except Exception as e:
            logger.error(f"Error scoring transaction: {e}")
            transactions_processed.labels(status='error').inc()
            raise


class RiskScoringService:
    """Main service class for risk scoring"""
    
    def __init__(self):
        self.running = True
        self.consumer = None
        self.producer = None
        self.fraud_engine = FraudDetectionEngine()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def initialize_kafka(self):
        """Initialize Kafka consumer and producer"""
        try:
            # Initialize consumer
            self.consumer = KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                max_poll_records=10,
                session_timeout_ms=30000
            )
            logger.info(f"Kafka consumer initialized for topic: {INPUT_TOPIC}")
            
            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info(f"Kafka producer initialized for topic: {OUTPUT_TOPIC}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    def process_transaction(self, transaction: Dict[str, Any]):
        """Process a single transaction"""
        active_transactions.inc()
        
        try:
            # Score the transaction
            scored_transaction = self.fraud_engine.calculate_fraud_score(transaction)
            
            # Publish to output topic
            future = self.producer.send(OUTPUT_TOPIC, value=scored_transaction)
            record_metadata = future.get(timeout=10)
            
            logger.info(
                f"Scored transaction published to {record_metadata.topic} "
                f"partition {record_metadata.partition} offset {record_metadata.offset}"
            )
            
            transactions_processed.labels(status='success').inc()
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            transactions_processed.labels(status='error').inc()
            raise
        finally:
            active_transactions.dec()
    
    def run(self):
        """Main service loop"""
        logger.info("Starting Risk Scoring Service...")
        
        # Start Prometheus metrics server
        start_http_server(METRICS_PORT)
        logger.info(f"Metrics server started on port {METRICS_PORT}")
        
        # Initialize Kafka
        self.initialize_kafka()
        
        logger.info("Service is ready to process transactions")
        
        try:
            # Main processing loop
            for message in self.consumer:
                if not self.running:
                    break
                
                try:
                    transaction = message.value
                    logger.info(f"Received transaction: {transaction.get('transaction_id')}")
                    self.process_transaction(transaction)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed")
        
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")
        
        logger.info("Shutdown complete")


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("FraudGuard-360 Risk Scoring Service")
    logger.info("=" * 60)
    
    service = RiskScoringService()
    
    try:
        service.run()
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
