"""
Integration Tests for API Gateway
==================================

Integration tests for API endpoints and Kafka integration.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json


# Import the app
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app


@pytest.fixture
def client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
def mock_kafka_producer():
    """Mock Kafka producer for testing"""
    with patch('app.get_kafka_producer') as mock:
        producer = MagicMock()
        future = MagicMock()
        future.get.return_value = MagicMock(
            topic='raw-transactions',
            partition=0,
            offset=123
        )
        producer.send.return_value = future
        mock.return_value = producer
        yield producer


class TestHealthEndpoints:
    """Test suite for health check endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root health check endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "api-gateway"
        assert "timestamp" in data
    
    def test_health_endpoint(self, client, mock_kafka_producer):
        """Test detailed health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "checks" in data
        assert "kafka" in data["checks"]


class TestTransactionEndpoint:
    """Test suite for transaction submission endpoint"""
    
    def test_create_transaction_success(self, client, mock_kafka_producer):
        """Test successful transaction submission"""
        transaction_data = {
            "user_id": "USR_12345",
            "amount": 250.50,
            "location": "New York, NY",
            "merchant_id": "MERCH_789",
            "transaction_type": "purchase"
        }
        
        response = client.post("/transactions", json=transaction_data)
        
        assert response.status_code == 202
        data = response.json()
        assert "transaction_id" in data
        assert data["status"] == "accepted"
        assert "Transaction submitted" in data["message"]
        assert "timestamp" in data
        
        # Verify transaction ID format
        assert data["transaction_id"].startswith("TXN_")
        assert transaction_data["user_id"] in data["transaction_id"]
    
    def test_create_transaction_minimal(self, client, mock_kafka_producer):
        """Test transaction with only required fields"""
        transaction_data = {
            "user_id": "USR_123",
            "amount": 100.0,
            "location": "Boston, MA"
        }
        
        response = client.post("/transactions", json=transaction_data)
        
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "accepted"
    
    def test_create_transaction_negative_amount(self, client):
        """Test transaction with negative amount (should fail validation)"""
        transaction_data = {
            "user_id": "USR_123",
            "amount": -50.0,
            "location": "Boston, MA"
        }
        
        response = client.post("/transactions", json=transaction_data)
        
        # Should fail validation
        assert response.status_code == 422  # Validation error
    
    def test_create_transaction_missing_required_field(self, client):
        """Test transaction with missing required fields"""
        transaction_data = {
            "user_id": "USR_123",
            "amount": 100.0
            # Missing 'location' field
        }
        
        response = client.post("/transactions", json=transaction_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_create_transaction_kafka_error(self, client):
        """Test transaction submission when Kafka is unavailable"""
        with patch('app.get_kafka_producer') as mock:
            mock.side_effect = Exception("Kafka connection failed")
            
            transaction_data = {
                "user_id": "USR_123",
                "amount": 100.0,
                "location": "Boston, MA"
            }
            
            response = client.post("/transactions", json=transaction_data)
            
            # Should return 503 Service Unavailable
            assert response.status_code == 503
            data = response.json()
            assert "unavailable" in data["detail"].lower()


class TestMetricsEndpoint:
    """Test suite for Prometheus metrics endpoint"""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns Prometheus format"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Prometheus metrics should be plain text
        assert "text/plain" in response.headers.get("content-type", "")


class TestTransactionStatusEndpoint:
    """Test suite for transaction status endpoint"""
    
    def test_get_transaction_status(self, client):
        """Test getting status of a transaction"""
        transaction_id = "TXN_12345_USR_123"
        
        response = client.get(f"/transactions/{transaction_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["transaction_id"] == transaction_id
        assert "status" in data
        assert "message" in data


class TestTransactionDataValidation:
    """Test suite for request data validation"""
    
    def test_transaction_with_all_fields(self, client, mock_kafka_producer):
        """Test transaction with all optional fields"""
        transaction_data = {
            "user_id": "USR_12345",
            "amount": 250.50,
            "location": "New York, NY",
            "merchant_id": "MERCH_789",
            "transaction_type": "purchase",
            "device_id": "DEV_456",
            "ip_address": "192.168.1.100"
        }
        
        response = client.post("/transactions", json=transaction_data)
        
        assert response.status_code == 202
    
    def test_transaction_amount_validation(self, client):
        """Test amount field validation"""
        # Zero amount should be valid
        transaction_data = {
            "user_id": "USR_123",
            "amount": 0.0,
            "location": "Boston, MA"
        }
        
        response = client.post("/transactions", json=transaction_data)
        assert response.status_code == 202
    
    @pytest.mark.parametrize("invalid_amount", [-1, -0.01, -1000])
    def test_transaction_invalid_amounts(self, client, invalid_amount):
        """Parametrized test for invalid amounts"""
        transaction_data = {
            "user_id": "USR_123",
            "amount": invalid_amount,
            "location": "Boston, MA"
        }
        
        response = client.post("/transactions", json=transaction_data)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
