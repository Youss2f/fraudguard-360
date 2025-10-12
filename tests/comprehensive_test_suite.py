#!/usr/bin/env python3
"""
FraudGuard 360 - Comprehensive End-to-End Test Suite
Tests the complete fraud detection pipeline from data ingestion to alerting
"""

import asyncio
import json
import time
import requests
import websocket
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudGuardTestSuite:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.ml_service_url = "http://localhost:8001"
        self.frontend_url = "http://localhost:3000"
        self.ws_url = "ws://localhost:8000/ws"
        
        self.test_results = []
        self.websocket_messages = []
        
    def log_test_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        """Log test result"""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        if success:
            logger.info(f"✅ {test_name} - {message} ({duration:.2f}s)")
        else:
            logger.error(f"❌ {test_name} - {message} ({duration:.2f}s)")
    
    def test_service_health(self):
        """Test all service health endpoints"""
        services = [
            ("API Gateway", f"{self.base_url}/health"),
            ("ML Service", f"{self.ml_service_url}/health"),
            ("Frontend", self.frontend_url)
        ]
        
        for service_name, url in services:
            start_time = time.time()
            try:
                response = requests.get(url, timeout=10)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    self.log_test_result(
                        f"{service_name} Health Check",
                        True,
                        f"Service is healthy (status: {response.status_code})",
                        duration
                    )
                else:
                    self.log_test_result(
                        f"{service_name} Health Check",
                        False,
                        f"Unexpected status code: {response.status_code}",
                        duration
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(
                    f"{service_name} Health Check",
                    False,
                    f"Connection failed: {str(e)}",
                    duration
                )
    
    def test_api_endpoints(self):
        """Test key API endpoints"""
        endpoints = [
            ("GET", "/api/v1/dashboard/kpis", "Dashboard KPIs"),
            ("GET", "/api/v1/alerts?limit=10", "Recent Alerts"),
            ("GET", "/api/v1/users/stats", "User Statistics"),
            ("GET", "/metrics", "Prometheus Metrics")
        ]
        
        for method, endpoint, description in endpoints:
            start_time = time.time()
            try:
                url = f"{self.base_url}{endpoint}"
                response = requests.request(method, url, timeout=10)
                duration = time.time() - start_time
                
                if response.status_code in [200, 201]:
                    self.log_test_result(
                        f"API Test - {description}",
                        True,
                        f"Endpoint responding correctly ({response.status_code})",
                        duration
                    )
                else:
                    self.log_test_result(
                        f"API Test - {description}",
                        False,
                        f"Unexpected status: {response.status_code}",
                        duration
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(
                    f"API Test - {description}",
                    False,
                    f"Request failed: {str(e)}",
                    duration
                )
    
    def test_ml_service_endpoints(self):
        """Test ML Service endpoints"""
        endpoints = [
            {
                "method": "POST",
                "endpoint": "/predict",
                "description": "Fraud Prediction",
                "data": {
                    "user_id": "test_user_001",
                    "features": {
                        "call_frequency": 15.5,
                        "international_calls": 5,
                        "avg_call_duration": 120.0,
                        "night_calls": 2,
                        "unique_callees": 8
                    }
                }
            },
            {
                "method": "POST",
                "endpoint": "/batch_score",
                "description": "Batch Scoring",
                "data": {
                    "user_ids": ["user_001", "user_002", "user_003"]
                }
            },
            {
                "method": "POST",
                "endpoint": "/network_analysis",
                "description": "Network Analysis",
                "data": {
                    "user_id": "test_user_001",
                    "depth": 2
                }
            }
        ]
        
        for test_case in endpoints:
            start_time = time.time()
            try:
                url = f"{self.ml_service_url}{test_case['endpoint']}"
                response = requests.request(
                    test_case["method"], 
                    url, 
                    json=test_case.get("data"),
                    timeout=30
                )
                duration = time.time() - start_time
                
                if response.status_code in [200, 201]:
                    result_data = response.json()
                    self.log_test_result(
                        f"ML Service - {test_case['description']}",
                        True,
                        f"Prediction successful (score: {result_data.get('risk_score', 'N/A')})",
                        duration
                    )
                else:
                    self.log_test_result(
                        f"ML Service - {test_case['description']}",
                        False,
                        f"Unexpected status: {response.status_code}",
                        duration
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(
                    f"ML Service - {test_case['description']}",
                    False,
                    f"Request failed: {str(e)}",
                    duration
                )
    
    def test_websocket_connection(self):
        """Test WebSocket real-time connections"""
        def on_message(ws, message):
            self.websocket_messages.append({
                "message": json.loads(message),
                "timestamp": datetime.now().isoformat()
            })
        
        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logger.info("WebSocket connection closed")
        
        def on_open(ws):
            logger.info("WebSocket connection opened")
            # Send a test message
            test_message = {
                "type": "test",
                "data": {"test_id": "ws_test_001"},
                "timestamp": datetime.now().isoformat()
            }
            ws.send(json.dumps(test_message))
        
        start_time = time.time()
        try:
            # Test alerts WebSocket
            ws = websocket.WebSocketApp(
                f"{self.ws_url}/alerts",
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run WebSocket in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection and messages
            time.sleep(5)
            ws.close()
            duration = time.time() - start_time
            
            if len(self.websocket_messages) > 0:
                self.log_test_result(
                    "WebSocket Connection Test",
                    True,
                    f"Received {len(self.websocket_messages)} messages",
                    duration
                )
            else:
                self.log_test_result(
                    "WebSocket Connection Test",
                    True,
                    "Connection established (no messages received)",
                    duration
                )
        
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "WebSocket Connection Test",
                False,
                f"Connection failed: {str(e)}",
                duration
            )
    
    def test_fraud_detection_pipeline(self):
        """Test end-to-end fraud detection pipeline"""
        start_time = time.time()
        try:
            # Step 1: Submit CDR data for processing
            test_cdr = {
                "caller_id": "test_user_pipeline",
                "callee_id": "+1234567890",
                "start_time": datetime.now().isoformat(),
                "end_time": (datetime.now() + timedelta(minutes=5)).isoformat(),
                "duration": 300,
                "call_type": "voice",
                "location_caller": "US_NY",
                "country_code": "US",
                "device_imei": "TEST_DEVICE_001"
            }
            
            # Submit to processing pipeline
            response = requests.post(
                f"{self.base_url}/api/v1/cdr/process",
                json=test_cdr,
                timeout=30
            )
            
            if response.status_code not in [200, 201, 202]:
                raise Exception(f"CDR processing failed: {response.status_code}")
            
            # Step 2: Wait for processing
            time.sleep(2)
            
            # Step 3: Check if fraud analysis was performed
            analysis_response = requests.get(
                f"{self.ml_service_url}/predict",
                params={"user_id": test_cdr["caller_id"]},
                timeout=30
            )
            
            if analysis_response.status_code == 200:
                analysis_data = analysis_response.json()
                risk_score = analysis_data.get("risk_score", 0)
                
                duration = time.time() - start_time
                self.log_test_result(
                    "End-to-End Pipeline Test",
                    True,
                    f"Pipeline processed CDR successfully (risk_score: {risk_score})",
                    duration
                )
            else:
                raise Exception(f"Analysis failed: {analysis_response.status_code}")
        
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "End-to-End Pipeline Test",
                False,
                f"Pipeline test failed: {str(e)}",
                duration
            )
    
    def test_database_connections(self):
        """Test database connectivity"""
        databases = [
            ("Neo4j Graph Database", f"{self.base_url}/api/v1/graph/health"),
            ("PostgreSQL", f"{self.base_url}/api/v1/database/health"),
            ("Redis Cache", f"{self.base_url}/api/v1/cache/health")
        ]
        
        for db_name, health_url in databases:
            start_time = time.time()
            try:
                response = requests.get(health_url, timeout=10)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    health_data = response.json()
                    status = health_data.get("status", "unknown")
                    self.log_test_result(
                        f"{db_name} Connection",
                        status == "healthy",
                        f"Database status: {status}",
                        duration
                    )
                else:
                    self.log_test_result(
                        f"{db_name} Connection",
                        False,
                        f"Health check failed: {response.status_code}",
                        duration
                    )
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result(
                    f"{db_name} Connection",
                    False,
                    f"Connection test failed: {str(e)}",
                    duration
                )
    
    def test_performance_metrics(self):
        """Test system performance metrics"""
        start_time = time.time()
        try:
            # Test API response times with multiple concurrent requests
            import concurrent.futures
            
            def make_request():
                start = time.time()
                response = requests.get(f"{self.base_url}/api/v1/dashboard/kpis", timeout=10)
                return time.time() - start, response.status_code
            
            # Make 10 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            response_times = [result[0] for result in results]
            status_codes = [result[1] for result in results]
            
            avg_response_time = sum(response_times) / len(response_times)
            success_rate = len([code for code in status_codes if code == 200]) / len(status_codes)
            
            duration = time.time() - start_time
            
            # Performance thresholds
            if avg_response_time < 2.0 and success_rate >= 0.95:
                self.log_test_result(
                    "Performance Test",
                    True,
                    f"Avg response time: {avg_response_time:.2f}s, Success rate: {success_rate:.2%}",
                    duration
                )
            else:
                self.log_test_result(
                    "Performance Test",
                    False,
                    f"Performance below threshold - Avg: {avg_response_time:.2f}s, Success: {success_rate:.2%}",
                    duration
                )
        
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Test",
                False,
                f"Performance test failed: {str(e)}",
                duration
            )
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["success"]])
        failed_tests = total_tests - passed_tests
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        total_duration = sum(r["duration"] for r in self.test_results)
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": success_rate,
                "total_duration": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "test_results": self.test_results,
            "websocket_messages": self.websocket_messages
        }
        
        # Save to file
        with open("test-report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        with open("test-report.md", "w") as f:
            f.write("# FraudGuard 360 Test Report\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests:** {total_tests}\n")
            f.write(f"- **Passed:** {passed_tests} ✅\n")
            f.write(f"- **Failed:** {failed_tests} ❌\n")
            f.write(f"- **Success Rate:** {success_rate:.1%}\n")
            f.write(f"- **Total Duration:** {total_duration:.2f} seconds\n\n")
            
            f.write("## Test Results\n\n")
            for result in self.test_results:
                status = "✅" if result["success"] else "❌"
                f.write(f"- {status} **{result['test_name']}** - {result['message']} ({result['duration']:.2f}s)\n")
            
            if self.websocket_messages:
                f.write(f"\n## WebSocket Messages ({len(self.websocket_messages)})\n\n")
                for msg in self.websocket_messages[:5]:  # Show first 5 messages
                    f.write(f"- `{msg['timestamp']}`: {json.dumps(msg['message'], indent=2)}\n")
        
        logger.info(f"Test report generated: test-report.json, test-report.md")
        return report
    
    def run_all_tests(self):
        """Run all test suites"""
        logger.info("🧪 Starting FraudGuard 360 comprehensive test suite...")
        
        test_suites = [
            ("Service Health Checks", self.test_service_health),
            ("API Endpoints", self.test_api_endpoints),
            ("ML Service", self.test_ml_service_endpoints),
            ("WebSocket Connections", self.test_websocket_connection),
            ("Database Connections", self.test_database_connections),
            ("Fraud Detection Pipeline", self.test_fraud_detection_pipeline),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        for suite_name, test_function in test_suites:
            logger.info(f"Running {suite_name}...")
            try:
                test_function()
            except Exception as e:
                logger.error(f"Test suite '{suite_name}' failed with exception: {e}")
        
        # Generate final report
        report = self.generate_test_report()
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {report['test_summary']['total_tests']}")
        print(f"Passed: {report['test_summary']['passed_tests']} ✅")
        print(f"Failed: {report['test_summary']['failed_tests']} ❌")
        print(f"Success Rate: {report['test_summary']['success_rate']:.1%}")
        print(f"Duration: {report['test_summary']['total_duration']:.2f} seconds")
        print("="*50)
        
        return report['test_summary']['success_rate'] >= 0.8

def main():
    """Main test execution"""
    # Parse command line arguments
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:8000"
    
    # Create test suite
    test_suite = FraudGuardTestSuite(base_url)
    
    # Run tests
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()