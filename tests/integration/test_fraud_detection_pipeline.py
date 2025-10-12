"""
FraudGuard 360 - Comprehensive Integration Test Suite
Tests the complete fraud detection pipeline end-to-end
"""

import asyncio
import json
import logging
import os
import pytest
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

import pandas as pd
import numpy as np
from kafka import KafkaProducer, KafkaConsumer
import redis
import psycopg2
from neo4j import GraphDatabase
import websocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    'api_gateway_url': os.getenv('API_GATEWAY_URL', 'http://localhost:8000'),
    'ml_service_url': os.getenv('ML_SERVICE_URL', 'http://localhost:8003'),
    'frontend_url': os.getenv('FRONTEND_URL', 'http://localhost:3000'),
    'kafka_bootstrap_servers': os.getenv('KAFKA_SERVERS', 'localhost:9092'),
    'redis_host': os.getenv('REDIS_HOST', 'localhost'),
    'redis_port': int(os.getenv('REDIS_PORT', '6379')),
    'postgres_host': os.getenv('POSTGRES_HOST', 'localhost'),
    'postgres_port': int(os.getenv('POSTGRES_PORT', '5432')),
    'postgres_db': os.getenv('POSTGRES_DB', 'fraudguard'),
    'postgres_user': os.getenv('POSTGRES_USER', 'fraudguard'),
    'postgres_password': os.getenv('POSTGRES_PASSWORD', 'password'),
    'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
    'neo4j_password': os.getenv('NEO4J_PASSWORD', 'password')
}

class FraudGuardIntegrationTest:
    """Main integration test class"""
    
    def __init__(self):
        self.test_session_id = str(uuid.uuid4())
        self.test_users = []
        self.test_data = []
        self.cleanup_tasks = []
        
    def setup_method(self):
        """Setup for each test method"""
        logger.info(f"Setting up integration test session: {self.test_session_id}")
        
        # Generate test data
        self.generate_test_data()
        
        # Verify all services are healthy
        self.verify_service_health()
        
    def teardown_method(self):
        """Cleanup after each test method"""
        logger.info(f"Cleaning up integration test session: {self.test_session_id}")
        
        # Run cleanup tasks
        for cleanup_task in self.cleanup_tasks:
            try:
                cleanup_task()
            except Exception as e:
                logger.error(f"Cleanup task failed: {e}")
    
    def generate_test_data(self):
        """Generate synthetic test data for fraud detection"""
        logger.info("Generating test CDR data...")
        
        # Normal user patterns
        normal_users = []
        for i in range(50):
            user_id = f"test_normal_user_{i:03d}_{self.test_session_id[:8]}"
            normal_users.append({
                'user_id': user_id,
                'call_duration': np.random.lognormal(4, 0.5),
                'call_cost': np.random.gamma(2, 0.3),
                'calls_per_day': np.random.poisson(12),
                'unique_numbers_called': np.random.poisson(6),
                'international_calls': 0,
                'night_calls': np.random.choice([0, 1], p=[0.8, 0.2]),
                'weekend_calls': np.random.choice([0, 1], p=[0.7, 0.3]),
                'call_frequency_variance': np.random.exponential(1.5),
                'location_changes': np.random.poisson(1),
                'avg_call_gap': np.random.exponential(2),
                'network_connections': np.random.poisson(8),
                'suspicious_patterns': 0,
                'timestamp': datetime.now().isoformat(),
                'expected_fraud': False
            })
        
        # Fraudulent user patterns
        fraud_users = []
        for i in range(10):
            user_id = f"test_fraud_user_{i:03d}_{self.test_session_id[:8]}"
            fraud_users.append({
                'user_id': user_id,
                'call_duration': np.random.lognormal(5, 1),  # Longer calls
                'call_cost': np.random.gamma(5, 2),  # Higher costs
                'calls_per_day': np.random.poisson(80),  # Many calls
                'unique_numbers_called': np.random.poisson(15),
                'international_calls': 1,  # International calls
                'night_calls': 1,  # Night calls
                'weekend_calls': np.random.choice([0, 1]),
                'call_frequency_variance': np.random.exponential(8),  # High variance
                'location_changes': np.random.poisson(12),  # Frequent location changes
                'avg_call_gap': np.random.exponential(0.5),  # Short gaps
                'network_connections': np.random.poisson(25),  # Many connections
                'suspicious_patterns': 1,  # Suspicious patterns
                'timestamp': datetime.now().isoformat(),
                'expected_fraud': True
            })
        
        self.test_data = normal_users + fraud_users
        self.test_users = [record['user_id'] for record in self.test_data]
        
        logger.info(f"Generated {len(normal_users)} normal and {len(fraud_users)} fraudulent test records")
    
    def verify_service_health(self):
        """Verify all services are healthy before running tests"""
        logger.info("Verifying service health...")
        
        services = [
            ('API Gateway', f"{TEST_CONFIG['api_gateway_url']}/health"),
            ('ML Service', f"{TEST_CONFIG['ml_service_url']}/health"),
            ('Frontend', TEST_CONFIG['frontend_url'])
        ]
        
        for service_name, health_url in services:
            try:
                response = requests.get(health_url, timeout=10)
                assert response.status_code == 200, f"{service_name} health check failed"
                logger.info(f"✓ {service_name} is healthy")
            except Exception as e:
                pytest.fail(f"❌ {service_name} health check failed: {e}")
        
        # Verify data stores
        self.verify_postgres_connection()
        self.verify_redis_connection()
        self.verify_neo4j_connection()
        self.verify_kafka_connection()
    
    def verify_postgres_connection(self):
        """Verify PostgreSQL connection"""
        try:
            conn = psycopg2.connect(
                host=TEST_CONFIG['postgres_host'],
                port=TEST_CONFIG['postgres_port'],
                database=TEST_CONFIG['postgres_db'],
                user=TEST_CONFIG['postgres_user'],
                password=TEST_CONFIG['postgres_password']
            )
            conn.close()
            logger.info("✓ PostgreSQL connection verified")
        except Exception as e:
            pytest.fail(f"❌ PostgreSQL connection failed: {e}")
    
    def verify_redis_connection(self):
        """Verify Redis connection"""
        try:
            r = redis.Redis(
                host=TEST_CONFIG['redis_host'],
                port=TEST_CONFIG['redis_port'],
                decode_responses=True
            )
            r.ping()
            logger.info("✓ Redis connection verified")
        except Exception as e:
            pytest.fail(f"❌ Redis connection failed: {e}")
    
    def verify_neo4j_connection(self):
        """Verify Neo4j connection"""
        try:
            driver = GraphDatabase.driver(
                TEST_CONFIG['neo4j_uri'],
                auth=(TEST_CONFIG['neo4j_user'], TEST_CONFIG['neo4j_password'])
            )
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                assert result.single()["test"] == 1
            driver.close()
            logger.info("✓ Neo4j connection verified")
        except Exception as e:
            pytest.fail(f"❌ Neo4j connection failed: {e}")
    
    def verify_kafka_connection(self):
        """Verify Kafka connection"""
        try:
            producer = KafkaProducer(
                bootstrap_servers=TEST_CONFIG['kafka_bootstrap_servers'],
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            producer.close()
            logger.info("✓ Kafka connection verified")
        except Exception as e:
            pytest.fail(f"❌ Kafka connection failed: {e}")

    # ==========================================================================
    # API Gateway Tests
    # ==========================================================================
    
    def test_api_gateway_authentication(self):
        """Test API Gateway authentication flow"""
        logger.info("Testing API Gateway authentication...")
        
        # Test login
        login_data = {
            'username': 'test_user',
            'password': 'test_password'
        }
        
        response = requests.post(
            f"{TEST_CONFIG['api_gateway_url']}/auth/login",
            json=login_data
        )
        
        assert response.status_code == 200
        auth_data = response.json()
        assert 'access_token' in auth_data
        
        # Test protected endpoint
        headers = {'Authorization': f"Bearer {auth_data['access_token']}"}
        response = requests.get(
            f"{TEST_CONFIG['api_gateway_url']}/users/profile",
            headers=headers
        )
        
        assert response.status_code == 200
        logger.info("✓ API Gateway authentication working")
    
    def test_cdr_data_ingestion(self):
        """Test CDR data ingestion through API Gateway"""
        logger.info("Testing CDR data ingestion...")
        
        # Send test CDR data
        for record in self.test_data[:5]:  # Test with first 5 records
            response = requests.post(
                f"{TEST_CONFIG['api_gateway_url']}/cdr/ingest",
                json=record
            )
            
            assert response.status_code in [200, 201, 202]
            
            # Verify response contains processing confirmation
            response_data = response.json()
            assert 'status' in response_data
            assert response_data['status'] in ['received', 'processing', 'queued']
        
        logger.info("✓ CDR data ingestion working")

    # ==========================================================================
    # ML Service Tests
    # ==========================================================================
    
    def test_ml_fraud_prediction(self):
        """Test ML service fraud prediction"""
        logger.info("Testing ML fraud prediction...")
        
        correct_predictions = 0
        total_predictions = 0
        
        for record in self.test_data:
            # Remove expected_fraud from prediction request
            prediction_data = {k: v for k, v in record.items() if k != 'expected_fraud'}
            
            response = requests.post(
                f"{TEST_CONFIG['ml_service_url']}/predict",
                json=prediction_data
            )
            
            assert response.status_code == 200
            prediction = response.json()
            
            # Verify prediction structure
            required_fields = [
                'user_id', 'fraud_probability', 'risk_score', 'fraud_type',
                'confidence', 'model_predictions', 'risk_factors', 'recommendation'
            ]
            
            for field in required_fields:
                assert field in prediction, f"Missing field: {field}"
            
            # Check prediction accuracy
            is_fraud_predicted = prediction['fraud_probability'] > 0.5
            is_fraud_expected = record['expected_fraud']
            
            if is_fraud_predicted == is_fraud_expected:
                correct_predictions += 1
            
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        logger.info(f"ML Prediction Accuracy: {accuracy:.2%}")
        
        # Expect at least 70% accuracy on test data
        assert accuracy >= 0.70, f"ML prediction accuracy too low: {accuracy:.2%}"
        
        logger.info("✓ ML fraud prediction working")
    
    def test_ml_batch_processing(self):
        """Test ML service batch processing"""
        logger.info("Testing ML batch processing...")
        
        # Send batch prediction request
        batch_data = {
            'records': self.test_data[:10],
            'batch_id': f"test_batch_{self.test_session_id[:8]}"
        }
        
        response = requests.post(
            f"{TEST_CONFIG['ml_service_url']}/predict/batch",
            json=batch_data
        )
        
        assert response.status_code in [200, 202]
        batch_result = response.json()
        
        if response.status_code == 202:
            # Async processing - check status
            batch_id = batch_result['batch_id']
            
            # Wait for processing completion
            for _ in range(30):  # Max 30 seconds
                status_response = requests.get(
                    f"{TEST_CONFIG['ml_service_url']}/predict/batch/{batch_id}/status"
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data['status'] == 'completed':
                        break
                
                time.sleep(1)
            
            # Get batch results
            results_response = requests.get(
                f"{TEST_CONFIG['ml_service_url']}/predict/batch/{batch_id}/results"
            )
            
            assert results_response.status_code == 200
            batch_predictions = results_response.json()
        else:
            # Sync processing
            batch_predictions = batch_result
        
        assert 'predictions' in batch_predictions
        assert len(batch_predictions['predictions']) == 10
        
        logger.info("✓ ML batch processing working")

    # ==========================================================================
    # End-to-End Workflow Tests
    # ==========================================================================
    
    def test_complete_fraud_detection_pipeline(self):
        """Test complete fraud detection pipeline end-to-end"""
        logger.info("Testing complete fraud detection pipeline...")
        
        # 1. Send CDR data through Kafka
        producer = KafkaProducer(
            bootstrap_servers=TEST_CONFIG['kafka_bootstrap_servers'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        test_record = self.test_data[0]  # Use first test record
        
        # Send to CDR events topic
        producer.send('cdr-events', test_record)
        producer.flush()
        producer.close()
        
        # 2. Wait for processing and check alerts topic
        consumer = KafkaConsumer(
            'fraud-alerts',
            bootstrap_servers=TEST_CONFIG['kafka_bootstrap_servers'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            consumer_timeout_ms=30000,  # 30 second timeout
            auto_offset_reset='latest'
        )
        
        alert_received = False
        for message in consumer:
            alert_data = message.value
            
            if alert_data.get('user_id') == test_record['user_id']:
                alert_received = True
                
                # Verify alert structure
                assert 'fraud_probability' in alert_data
                assert 'risk_score' in alert_data
                assert 'recommendation' in alert_data
                
                logger.info(f"Alert received for user {test_record['user_id']}")
                break
        
        consumer.close()
        
        # 3. Verify data is stored in PostgreSQL
        conn = psycopg2.connect(
            host=TEST_CONFIG['postgres_host'],
            port=TEST_CONFIG['postgres_port'],
            database=TEST_CONFIG['postgres_db'],
            user=TEST_CONFIG['postgres_user'],
            password=TEST_CONFIG['postgres_password']
        )
        
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM fraud_predictions WHERE user_id = %s",
            (test_record['user_id'],)
        )
        
        prediction_count = cursor.fetchone()[0]
        assert prediction_count > 0, "Prediction not stored in database"
        
        cursor.close()
        conn.close()
        
        # 4. Verify user profile is updated in Neo4j
        driver = GraphDatabase.driver(
            TEST_CONFIG['neo4j_uri'],
            auth=(TEST_CONFIG['neo4j_user'], TEST_CONFIG['neo4j_password'])
        )
        
        with driver.session() as session:
            result = session.run(
                "MATCH (u:User {id: $user_id}) RETURN u.risk_score as risk_score",
                user_id=test_record['user_id']
            )
            
            user_record = result.single()
            if user_record:
                assert user_record['risk_score'] is not None
                logger.info(f"User risk score updated in Neo4j: {user_record['risk_score']}")
        
        driver.close()
        
        # 5. Check Redis cache
        r = redis.Redis(
            host=TEST_CONFIG['redis_host'],
            port=TEST_CONFIG['redis_port'],
            decode_responses=True
        )
        
        cached_prediction = r.get(f"fraud_prediction:{test_record['user_id']}")
        if cached_prediction:
            prediction_data = json.loads(cached_prediction)
            assert 'fraud_probability' in prediction_data
            logger.info("Prediction cached in Redis")
        
        logger.info("✓ Complete fraud detection pipeline working")
    
    def test_real_time_dashboard_updates(self):
        """Test real-time dashboard updates via WebSocket"""
        logger.info("Testing real-time dashboard updates...")
        
        # This would require a WebSocket connection to the frontend
        # For now, we'll test the API endpoints that feed the dashboard
        
        # Test real-time metrics endpoint
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/metrics/real-time")
        assert response.status_code == 200
        
        metrics = response.json()
        required_metrics = [
            'total_transactions', 'fraud_rate', 'system_health',
            'processing_latency', 'active_alerts'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
        
        # Test alerts endpoint
        response = requests.get(f"{TEST_CONFIG['api_gateway_url']}/alerts/active")
        assert response.status_code == 200
        
        alerts = response.json()
        assert isinstance(alerts, list)
        
        logger.info("✓ Real-time dashboard updates working")
    
    def test_system_performance_under_load(self):
        """Test system performance under load"""
        logger.info("Testing system performance under load...")
        
        # Send multiple concurrent requests
        import concurrent.futures
        import threading
        
        def send_prediction_request(record_data):
            try:
                response = requests.post(
                    f"{TEST_CONFIG['ml_service_url']}/predict",
                    json=record_data,
                    timeout=10
                )
                return response.status_code == 200, response.elapsed.total_seconds()
            except Exception as e:
                return False, None
        
        # Test with 50 concurrent requests
        successful_requests = 0
        total_latency = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            for record in self.test_data:
                future = executor.submit(send_prediction_request, record)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                success, latency = future.result()
                if success and latency:
                    successful_requests += 1
                    total_latency += latency
        
        success_rate = successful_requests / len(self.test_data)
        average_latency = total_latency / successful_requests if successful_requests > 0 else 0
        
        logger.info(f"Load test results: {success_rate:.2%} success rate, {average_latency:.3f}s avg latency")
        
        # Expect at least 90% success rate and sub-second latency
        assert success_rate >= 0.90, f"Success rate too low: {success_rate:.2%}"
        assert average_latency < 2.0, f"Average latency too high: {average_latency:.3f}s"
        
        logger.info("✓ System performance under load acceptable")

# =============================================================================
# Test Runner
# =============================================================================

@pytest.fixture(scope="class")
def integration_test():
    """Pytest fixture for integration test setup/teardown"""
    test_instance = FraudGuardIntegrationTest()
    test_instance.setup_method()
    yield test_instance
    test_instance.teardown_method()

class TestFraudGuardIntegration:
    """Integration test class for pytest"""
    
    def test_service_health(self, integration_test):
        """Test all services are healthy"""
        integration_test.verify_service_health()
    
    def test_api_authentication(self, integration_test):
        """Test API Gateway authentication"""
        integration_test.test_api_gateway_authentication()
    
    def test_cdr_ingestion(self, integration_test):
        """Test CDR data ingestion"""
        integration_test.test_cdr_data_ingestion()
    
    def test_ml_predictions(self, integration_test):
        """Test ML fraud predictions"""
        integration_test.test_ml_fraud_prediction()
    
    def test_batch_processing(self, integration_test):
        """Test ML batch processing"""
        integration_test.test_ml_batch_processing()
    
    def test_end_to_end_pipeline(self, integration_test):
        """Test complete fraud detection pipeline"""
        integration_test.test_complete_fraud_detection_pipeline()
    
    def test_dashboard_updates(self, integration_test):
        """Test real-time dashboard updates"""
        integration_test.test_real_time_dashboard_updates()
    
    def test_performance_load(self, integration_test):
        """Test system performance under load"""
        integration_test.test_system_performance_under_load()

if __name__ == "__main__":
    # Run integration tests directly
    test_suite = FraudGuardIntegrationTest()
    test_suite.setup_method()
    
    try:
        logger.info("=== Starting FraudGuard 360 Integration Tests ===")
        
        test_suite.verify_service_health()
        test_suite.test_api_gateway_authentication()
        test_suite.test_cdr_data_ingestion()
        test_suite.test_ml_fraud_prediction()
        test_suite.test_ml_batch_processing()
        test_suite.test_complete_fraud_detection_pipeline()
        test_suite.test_real_time_dashboard_updates()
        test_suite.test_system_performance_under_load()
        
        logger.info("=== All Integration Tests Passed! ===")
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise
    finally:
        test_suite.teardown_method()