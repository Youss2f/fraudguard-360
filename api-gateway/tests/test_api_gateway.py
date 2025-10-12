import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

# Import the FastAPI app and dependencies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.monitoring.metrics import prometheus_metrics
from app.monitoring.health import HealthChecker


class TestAPIGateway:
    """Comprehensive unit tests for API Gateway service"""
    
    def setup_method(self):
        """Setup test client and mocks"""
        self.client = TestClient(app)
        self.mock_health_checker = Mock(spec=HealthChecker)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format"""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "FraudGuard 360 API Gateway"
        assert "version" in data
    
    @patch('app.main.httpx.AsyncClient.post')
    async def test_analyze_cdr_endpoint(self, mock_post):
        """Test CDR analysis endpoint"""
        # Mock ML service response
        mock_response = Mock()
        mock_response.json.return_value = {
            "fraud_score": 0.85,
            "fraud_type": "suspicious_pattern",
            "confidence": 0.92
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Test CDR data
        cdr_data = {
            "call_id": "test_call_123",
            "caller_number": "1234567890",
            "callee_number": "0987654321",
            "duration": 300,
            "cost": 5.50,
            "timestamp": datetime.now().isoformat()
        }
        
        response = self.client.post("/api/v1/analyze/cdr", json=cdr_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "fraud_score" in result
        assert "analysis_id" in result
        assert result["status"] == "completed"
    
    def test_analyze_cdr_invalid_data(self):
        """Test CDR analysis with invalid data"""
        invalid_data = {
            "call_id": "",  # Invalid empty call_id
            "duration": -1  # Invalid negative duration
        }
        
        response = self.client.post("/api/v1/analyze/cdr", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @patch('app.main.httpx.AsyncClient.get')
    async def test_get_alerts_endpoint(self, mock_get):
        """Test alerts retrieval endpoint"""
        # Mock database response
        mock_response = Mock()
        mock_response.json.return_value = {
            "alerts": [
                {
                    "id": "alert_123",
                    "fraud_score": 0.95,
                    "severity": "high",
                    "timestamp": datetime.now().isoformat(),
                    "description": "Suspicious calling pattern detected"
                }
            ],
            "total": 1
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        response = self.client.get("/api/v1/alerts")
        assert response.status_code == 200
        
        result = response.json()
        assert "alerts" in result
        assert "total" in result
    
    def test_get_alerts_with_filters(self):
        """Test alerts endpoint with query filters"""
        response = self.client.get("/api/v1/alerts?severity=high&limit=10")
        assert response.status_code == 200
    
    @patch('app.main.httpx.AsyncClient.get')
    async def test_network_data_endpoint(self, mock_get):
        """Test network visualization data endpoint"""
        # Mock Neo4j response
        mock_response = Mock()
        mock_response.json.return_value = {
            "nodes": [
                {"id": "node1", "label": "Phone", "properties": {"number": "1234567890"}},
                {"id": "node2", "label": "Phone", "properties": {"number": "0987654321"}}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "weight": 5, "fraud_score": 0.3}
            ]
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        response = self.client.get("/api/v1/network/visualization")
        assert response.status_code == 200
        
        result = response.json()
        assert "nodes" in result
        assert "edges" in result
    
    def test_api_versioning(self):
        """Test API versioning"""
        # Test v1 endpoints
        response = self.client.get("/api/v1/alerts")
        assert response.status_code == 200
        
        # Test that non-existent version returns 404
        response = self.client.get("/api/v2/alerts")
        assert response.status_code == 404
    
    def test_cors_headers(self):
        """Test CORS headers are properly set"""
        response = self.client.options("/api/v1/alerts")
        assert "access-control-allow-origin" in response.headers
    
    @patch('app.main.prometheus_metrics')
    def test_metrics_tracking(self, mock_metrics):
        """Test that metrics are properly tracked"""
        # Make a request to trigger metrics
        response = self.client.get("/health")
        assert response.status_code == 200
        
        # Verify metrics were called (would need actual implementation)
        # This is a placeholder for metrics verification
    
    def test_error_handling(self):
        """Test error responses and handling"""
        # Test 404 for non-existent endpoint
        response = self.client.get("/non-existent-endpoint")
        assert response.status_code == 404
        
        # Test proper error format
        assert "detail" in response.json()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        # Create multiple concurrent requests
        tasks = []
        for i in range(10):
            task = asyncio.create_task(self.make_async_request())
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        for result in results:
            assert result == 200
    
    async def make_async_request(self):
        """Helper method for async request testing"""
        response = self.client.get("/health")
        return response.status_code


class TestHealthChecker:
    """Unit tests for health checking functionality"""
    
    def setup_method(self):
        """Setup health checker instance"""
        self.health_checker = HealthChecker()
    
    @patch('asyncpg.connect')
    async def test_postgres_health_check(self, mock_connect):
        """Test PostgreSQL health check"""
        # Mock successful connection
        mock_conn = AsyncMock()
        mock_connect.return_value = mock_conn
        
        result = await self.health_checker.check_postgres()
        assert result.status == "healthy"
        assert result.response_time > 0
        
        # Test connection failure
        mock_connect.side_effect = Exception("Connection failed")
        result = await self.health_checker.check_postgres()
        assert result.status == "unhealthy"
    
    @patch('redis.asyncio.Redis')
    async def test_redis_health_check(self, mock_redis):
        """Test Redis health check"""
        # Mock successful ping
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = True
        mock_redis.return_value = mock_redis_instance
        
        result = await self.health_checker.check_redis()
        assert result.status == "healthy"
    
    @patch('httpx.AsyncClient.get')
    async def test_ml_service_health_check(self, mock_get):
        """Test ML service health check"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response
        
        result = await self.health_checker.check_ml_service()
        assert result.status == "healthy"
    
    async def test_system_health_calculation(self):
        """Test overall system health calculation"""
        # Mock individual component results
        with patch.object(self.health_checker, 'check_postgres') as mock_postgres, \
             patch.object(self.health_checker, 'check_redis') as mock_redis, \
             patch.object(self.health_checker, 'check_neo4j') as mock_neo4j:
            
            # All healthy
            mock_postgres.return_value = Mock(status="healthy")
            mock_redis.return_value = Mock(status="healthy")
            mock_neo4j.return_value = Mock(status="healthy")
            
            system_health = await self.health_checker.get_system_health()
            assert system_health.overall_status == "healthy"
            
            # One unhealthy
            mock_postgres.return_value = Mock(status="unhealthy")
            system_health = await self.health_checker.get_system_health()
            assert system_health.overall_status == "degraded"


class TestMetricsCollection:
    """Unit tests for metrics collection functionality"""
    
    def setup_method(self):
        """Setup metrics collection tests"""
        self.test_client = TestClient(app)
    
    def test_api_request_metrics(self):
        """Test API request metrics are collected"""
        with patch('app.monitoring.metrics.API_REQUESTS_TOTAL') as mock_counter:
            response = self.test_client.get("/health")
            assert response.status_code == 200
            
            # Verify counter was incremented (would need actual implementation)
            # This is a placeholder for metrics verification
    
    def test_fraud_detection_metrics(self):
        """Test fraud detection metrics are collected"""
        with patch('app.monitoring.metrics.FRAUD_DETECTIONS_TOTAL') as mock_counter:
            # Would test actual fraud detection metric recording
            pass
    
    def test_response_time_metrics(self):
        """Test response time metrics are collected"""
        with patch('app.monitoring.metrics.API_REQUEST_DURATION') as mock_histogram:
            response = self.test_client.get("/health")
            assert response.status_code == 200
            
            # Verify histogram was updated (would need actual implementation)
            # This is a placeholder for metrics verification


if __name__ == "__main__":
    pytest.main(["-v", __file__])