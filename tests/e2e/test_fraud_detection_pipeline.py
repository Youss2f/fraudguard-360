"""
End-to-End Test Suite for FraudGuard 360° Fraud Detection Pipeline

This module tests the complete fraud detection workflow from CDR ingestion
to fraud alert generation and visualization.
"""

import asyncio
import json
import time
from typing import Dict, List, Any
import requests
import pytest
from kafka import KafkaProducer
from neo4j import GraphDatabase
import websocket


class FraudDetectionE2ETests:
    """Comprehensive end-to-end tests for fraud detection pipeline."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.ml_service_url = "http://localhost:5000"
        self.frontend_url = "http://localhost:3000"
        self.neo4j_uri = "bolt://localhost:7687"
        self.kafka_bootstrap_servers = "localhost:9092"
        
    def setup_method(self):
        """Set up test environment before each test."""
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri, 
            auth=("neo4j", "password")
        )
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=[self.kafka_bootstrap_servers],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    def teardown_method(self):
        """Clean up test environment after each test."""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()
        if hasattr(self, 'kafka_producer'):
            self.kafka_producer.close()

    @pytest.mark.asyncio
    async def test_complete_fraud_detection_pipeline(self):
        """
        Test the complete fraud detection pipeline:
        1. Ingest CDR data via Kafka
        2. Process data through Flink
        3. Store in Neo4j
        4. Trigger ML analysis
        5. Generate fraud alerts
        6. Display in frontend
        """
        # Step 1: Generate test CDR data
        test_cdr = {
            "call_id": "test_call_12345",
            "caller_id": "subscriber_001",
            "callee_id": "subscriber_002",
            "timestamp": int(time.time() * 1000),
            "duration": 120,
            "call_type": "voice",
            "location": "geo:40.7128,-74.0060",
            "suspicious_pattern": True  # Mark as potentially fraudulent
        }
        
        # Step 2: Send CDR to Kafka
        self.kafka_producer.send('cdr-events', test_cdr)
        self.kafka_producer.flush()
        
        # Step 3: Wait for Flink processing
        await asyncio.sleep(5)
        
        # Step 4: Verify data in Neo4j
        with self.neo4j_driver.session() as session:
            result = session.run(
                "MATCH (s:Subscriber {id: $caller_id}) RETURN s",
                caller_id=test_cdr["caller_id"]
            )
            assert result.single() is not None, "Subscriber not found in Neo4j"
        
        # Step 5: Trigger fraud detection analysis
        response = requests.post(
            f"{self.ml_service_url}/detect-fraud",
            json={"subscriber_id": test_cdr["caller_id"]}
        )
        assert response.status_code == 200
        fraud_result = response.json()
        
        # Step 6: Verify fraud detection results
        assert "fraud_score" in fraud_result
        assert "risk_level" in fraud_result
        assert fraud_result["fraud_score"] > 0.5  # Should detect as fraudulent
        
        # Step 7: Check fraud alert creation via API
        response = requests.get(f"{self.api_base_url}/alerts/recent")
        assert response.status_code == 200
        alerts = response.json()
        
        # Find our test alert
        test_alert = next(
            (alert for alert in alerts 
             if alert.get("subscriber_id") == test_cdr["caller_id"]), 
            None
        )
        assert test_alert is not None, "Fraud alert not created"
        assert test_alert["severity"] in ["medium", "high", "critical"]

    def test_real_time_dashboard_updates(self):
        """Test real-time updates in the fraud detection dashboard."""
        # Connect to WebSocket endpoint
        ws_url = "ws://localhost:8000/ws/fraud-alerts"
        
        received_alerts = []
        
        def on_message(ws, message):
            alert = json.loads(message)
            received_alerts.append(alert)
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error
        )
        
        # Start WebSocket in background
        import threading
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for connection
        time.sleep(2)
        
        # Generate fraud event
        fraud_cdr = {
            "call_id": "test_fraud_67890",
            "caller_id": "suspicious_user_001",
            "callee_id": "victim_user_002",
            "timestamp": int(time.time() * 1000),
            "duration": 300,
            "call_type": "voice",
            "location": "geo:25.7617,-80.1918",  # Different location
            "fraud_indicators": ["unusual_location", "high_frequency"]
        }
        
        # Send to Kafka
        self.kafka_producer.send('cdr-events', fraud_cdr)
        self.kafka_producer.flush()
        
        # Wait for processing and WebSocket update
        time.sleep(10)
        
        # Verify real-time alert received
        assert len(received_alerts) > 0, "No real-time alerts received"
        latest_alert = received_alerts[-1]
        assert latest_alert["subscriber_id"] == fraud_cdr["caller_id"]
        
        ws.close()

    def test_performance_under_load(self):
        """Test system performance under high CDR load."""
        # Generate batch of CDRs
        batch_size = 1000
        cdrs = []
        
        for i in range(batch_size):
            cdr = {
                "call_id": f"load_test_{i}",
                "caller_id": f"user_{i % 100}",  # 100 unique users
                "callee_id": f"user_{(i + 1) % 100}",
                "timestamp": int(time.time() * 1000) + i,
                "duration": 60 + (i % 240),  # 1-5 minutes
                "call_type": "voice",
                "location": f"geo:40.{7000 + i % 1000},-74.{i % 1000}"
            }
            cdrs.append(cdr)
        
        # Measure ingestion time
        start_time = time.time()
        
        for cdr in cdrs:
            self.kafka_producer.send('cdr-events', cdr)
        
        self.kafka_producer.flush()
        ingestion_time = time.time() - start_time
        
        # Verify performance metrics
        throughput = batch_size / ingestion_time
        assert throughput > 500, f"Throughput too low: {throughput} records/sec"
        
        # Wait for processing
        time.sleep(30)
        
        # Check if all records processed
        with self.neo4j_driver.session() as session:
            result = session.run(
                "MATCH (s:Subscriber) RETURN count(s) as count"
            )
            subscriber_count = result.single()["count"]
            assert subscriber_count >= 100, "Not all subscribers processed"

    def test_fraud_detection_accuracy(self):
        """Test ML model accuracy with known fraud patterns."""
        # Known fraud patterns for testing
        fraud_patterns = [
            {
                "pattern": "call_frequency_spike",
                "description": "Sudden spike in call frequency",
                "expected_fraud_score": 0.8
            },
            {
                "pattern": "unusual_location",
                "description": "Calls from unusual geographic location",
                "expected_fraud_score": 0.7
            },
            {
                "pattern": "short_duration_high_frequency",
                "description": "Many short duration calls",
                "expected_fraud_score": 0.9
            }
        ]
        
        for pattern in fraud_patterns:
            # Generate CDRs matching the fraud pattern
            if pattern["pattern"] == "call_frequency_spike":
                # Generate 50 calls in 5 minutes
                base_timestamp = int(time.time() * 1000)
                for i in range(50):
                    cdr = {
                        "call_id": f"freq_spike_{i}",
                        "caller_id": "fraud_user_frequency",
                        "callee_id": f"target_{i % 10}",
                        "timestamp": base_timestamp + (i * 6000),  # 6 seconds apart
                        "duration": 30,
                        "call_type": "voice"
                    }
                    self.kafka_producer.send('cdr-events', cdr)
            
            self.kafka_producer.flush()
            
            # Wait for processing
            time.sleep(10)
            
            # Test fraud detection
            response = requests.post(
                f"{self.ml_service_url}/detect-fraud",
                json={"subscriber_id": "fraud_user_frequency"}
            )
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["fraud_score"] >= pattern["expected_fraud_score"], \
                f"Low fraud score for {pattern['pattern']}: {result['fraud_score']}"

    def test_system_recovery_after_failure(self):
        """Test system recovery capabilities after service failures."""
        # Test database connection failure recovery
        # This would typically involve Docker container manipulation
        # or network simulation in a real test environment
        
        # For now, test API error handling
        response = requests.post(
            f"{self.api_base_url}/detect-fraud",
            json={"invalid": "data"}
        )
        assert response.status_code == 422  # Validation error
        
        # Test valid request after error
        response = requests.post(
            f"{self.api_base_url}/detect-fraud",
            json={"subscriber_id": "test_user"}
        )
        assert response.status_code in [200, 404]  # Valid response format

    def test_data_privacy_compliance(self):
        """Test data privacy and compliance features."""
        # Test data anonymization
        response = requests.get(
            f"{self.api_base_url}/analytics/patterns",
            params={"anonymize": True}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Verify no personal identifiers in anonymized data
        assert "caller_id" not in str(data), "Personal identifier found in anonymized data"
        assert "phone_number" not in str(data), "Phone number found in anonymized data"
        
        # Test data retention policies
        # This would check if old data is properly archived/deleted
        pass


if __name__ == "__main__":
    # Run specific test for demonstration
    test_suite = FraudDetectionE2ETests()
    test_suite.setup_method()
    
    try:
        asyncio.run(test_suite.test_complete_fraud_detection_pipeline())
        print("✅ End-to-end fraud detection pipeline test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        test_suite.teardown_method()