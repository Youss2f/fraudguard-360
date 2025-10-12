# FraudGuard 360 - Security Testing Configuration
# Comprehensive security testing and vulnerability assessment

import subprocess
import json
import os
import sys
import time
import requests
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security-testing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SecurityTester:
    def __init__(self, base_url="http://localhost:8000", results_dir="security-results"):
        self.base_url = base_url
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_all_tests(self):
        """Run comprehensive security test suite"""
        logger.info("🔒 Starting FraudGuard 360 Security Testing Suite")
        
        results = {
            "timestamp": self.timestamp,
            "base_url": self.base_url,
            "tests": {}
        }
        
        # Run all security tests
        test_methods = [
            ("input_validation", self.test_input_validation),
            ("authentication", self.test_authentication_security),
            ("authorization", self.test_authorization),
            ("sql_injection", self.test_sql_injection),
            ("xss_protection", self.test_xss_protection),
            ("csrf_protection", self.test_csrf_protection),
            ("rate_limiting", self.test_rate_limiting),
            ("data_exposure", self.test_sensitive_data_exposure),
            ("ssl_tls", self.test_ssl_tls_security),
            ("headers_security", self.test_security_headers),
            ("api_security", self.test_api_security),
            ("fraud_specific", self.test_fraud_detection_security)
        ]
        
        for test_name, test_method in test_methods:
            logger.info(f"🧪 Running {test_name} tests...")
            try:
                results["tests"][test_name] = test_method()
                logger.info(f"✅ {test_name} tests completed")
            except Exception as e:
                logger.error(f"❌ {test_name} tests failed: {str(e)}")
                results["tests"][test_name] = {"error": str(e), "status": "failed"}
        
        # Save results
        self.save_results(results)
        self.generate_report(results)
        
        return results
    
    def test_input_validation(self):
        """Test input validation and sanitization"""
        logger.info("Testing input validation...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Test CDR analysis endpoint with malicious inputs
        malicious_inputs = [
            # SQL injection attempts
            {"caller_number": "'; DROP TABLE users; --"},
            {"caller_number": "1' OR '1'='1"},
            {"callee_number": "1' UNION SELECT * FROM users--"},
            
            # XSS attempts
            {"caller_number": "<script>alert('xss')</script>"},
            {"callee_number": "javascript:alert('xss')"},
            
            # Command injection
            {"caller_number": "; cat /etc/passwd"},
            {"caller_number": "$(whoami)"},
            {"caller_number": "`id`"},
            
            # Path traversal
            {"caller_number": "../../../../etc/passwd"},
            {"caller_number": "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts"},
            
            # Buffer overflow attempts
            {"caller_number": "A" * 10000},
            {"duration": -1},
            {"cost": float('inf')},
            
            # NoSQL injection
            {"caller_number": {"$ne": None}},
            {"caller_number": {"$gt": ""}},
        ]
        
        for payload in malicious_inputs:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/analyze/cdr",
                    json=payload,
                    timeout=10
                )
                
                # Check if malicious input was processed successfully (vulnerability)
                if response.status_code == 200:
                    results["vulnerabilities"].append({
                        "type": "input_validation",
                        "payload": payload,
                        "response_code": response.status_code,
                        "severity": "high"
                    })
                    results["status"] = "vulnerable"
                
            except requests.RequestException:
                # Request failure is expected for malicious inputs
                pass
        
        return results
    
    def test_authentication_security(self):
        """Test authentication mechanisms"""
        logger.info("Testing authentication security...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Test endpoints without authentication
        protected_endpoints = [
            "/api/v1/admin/users",
            "/api/v1/admin/config",
            "/api/v1/alerts/manage",
        ]
        
        for endpoint in protected_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code != 401:
                    results["vulnerabilities"].append({
                        "type": "authentication_bypass",
                        "endpoint": endpoint,
                        "response_code": response.status_code,
                        "severity": "critical"
                    })
                    results["status"] = "vulnerable"
            except requests.RequestException:
                pass
        
        # Test weak authentication
        weak_credentials = [
            ("admin", "admin"),
            ("admin", "password"),
            ("admin", "123456"),
            ("root", "root"),
            ("test", "test")
        ]
        
        for username, password in weak_credentials:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/auth/login",
                    json={"username": username, "password": password},
                    timeout=10
                )
                if response.status_code == 200:
                    results["vulnerabilities"].append({
                        "type": "weak_credentials",
                        "credentials": f"{username}:{password}",
                        "severity": "critical"
                    })
                    results["status"] = "vulnerable"
            except requests.RequestException:
                pass
        
        return results
    
    def test_authorization(self):
        """Test authorization and access control"""
        logger.info("Testing authorization...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Test privilege escalation
        try:
            # Try to access admin endpoints with regular user token
            response = requests.get(
                f"{self.base_url}/api/v1/admin/users",
                headers={"Authorization": "Bearer fake_user_token"},
                timeout=10
            )
            if response.status_code == 200:
                results["vulnerabilities"].append({
                    "type": "privilege_escalation",
                    "endpoint": "/api/v1/admin/users",
                    "severity": "critical"
                })
                results["status"] = "vulnerable"
        except requests.RequestException:
            pass
        
        return results
    
    def test_sql_injection(self):
        """Test SQL injection vulnerabilities"""
        logger.info("Testing SQL injection...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # SQL injection payloads
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT version() --",
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "' OR 1=1#"
        ]
        
        # Test various endpoints
        endpoints = [
            ("/api/v1/alerts", "caller_number"),
            ("/api/v1/network/search", "phone_number"),
            ("/api/v1/reports/fraud", "user_id")
        ]
        
        for endpoint, param in endpoints:
            for payload in sql_payloads:
                try:
                    response = requests.get(
                        f"{self.base_url}{endpoint}",
                        params={param: payload},
                        timeout=10
                    )
                    
                    # Check for SQL error messages
                    if any(error in response.text.lower() for error in [
                        'sql syntax', 'mysql', 'postgresql', 'sqlite', 'ora-', 'syntax error'
                    ]):
                        results["vulnerabilities"].append({
                            "type": "sql_injection",
                            "endpoint": endpoint,
                            "payload": payload,
                            "severity": "critical"
                        })
                        results["status"] = "vulnerable"
                        
                except requests.RequestException:
                    pass
        
        return results
    
    def test_xss_protection(self):
        """Test Cross-Site Scripting protection"""
        logger.info("Testing XSS protection...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/analyze/cdr",
                    json={"caller_number": payload},
                    timeout=10
                )
                
                # Check if script tags are reflected without encoding
                if payload in response.text:
                    results["vulnerabilities"].append({
                        "type": "reflected_xss",
                        "payload": payload,
                        "severity": "high"
                    })
                    results["status"] = "vulnerable"
                    
            except requests.RequestException:
                pass
        
        return results
    
    def test_csrf_protection(self):
        """Test CSRF protection"""
        logger.info("Testing CSRF protection...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Test state-changing operations without CSRF tokens
        csrf_endpoints = [
            ("POST", "/api/v1/alerts/acknowledge"),
            ("DELETE", "/api/v1/alerts/1"),
            ("PUT", "/api/v1/config/rules"),
        ]
        
        for method, endpoint in csrf_endpoints:
            try:
                response = requests.request(
                    method,
                    f"{self.base_url}{endpoint}",
                    timeout=10
                )
                
                # If successful without CSRF token, it's vulnerable
                if response.status_code in [200, 201, 204]:
                    results["vulnerabilities"].append({
                        "type": "csrf_missing",
                        "endpoint": endpoint,
                        "method": method,
                        "severity": "medium"
                    })
                    results["status"] = "vulnerable"
                    
            except requests.RequestException:
                pass
        
        return results
    
    def test_rate_limiting(self):
        """Test rate limiting implementation"""
        logger.info("Testing rate limiting...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Test rapid requests to check for rate limiting
        endpoint = f"{self.base_url}/api/v1/analyze/cdr"
        payload = {"caller_number": "1234567890", "callee_number": "0987654321"}
        
        successful_requests = 0
        for i in range(100):  # Send 100 rapid requests
            try:
                response = requests.post(endpoint, json=payload, timeout=5)
                if response.status_code == 200:
                    successful_requests += 1
                elif response.status_code == 429:  # Rate limited
                    break
            except requests.RequestException:
                break
        
        # If too many requests succeeded, rate limiting might be weak
        if successful_requests > 50:
            results["vulnerabilities"].append({
                "type": "weak_rate_limiting",
                "successful_requests": successful_requests,
                "severity": "medium"
            })
            results["status"] = "vulnerable"
        
        return results
    
    def test_sensitive_data_exposure(self):
        """Test for sensitive data exposure"""
        logger.info("Testing sensitive data exposure...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Check common endpoints for data leakage
        endpoints = [
            "/api/v1/debug",
            "/api/v1/config",
            "/api/v1/status",
            "/.env",
            "/backup",
            "/admin"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    # Check for sensitive patterns in response
                    sensitive_patterns = [
                        'password', 'secret', 'token', 'key', 'database_url',
                        'aws_access_key', 'api_key', 'private_key'
                    ]
                    
                    for pattern in sensitive_patterns:
                        if pattern in response.text.lower():
                            results["vulnerabilities"].append({
                                "type": "sensitive_data_exposure",
                                "endpoint": endpoint,
                                "pattern": pattern,
                                "severity": "high"
                            })
                            results["status"] = "vulnerable"
                            
            except requests.RequestException:
                pass
        
        return results
    
    def test_ssl_tls_security(self):
        """Test SSL/TLS configuration"""
        logger.info("Testing SSL/TLS security...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        if self.base_url.startswith('https'):
            try:
                response = requests.get(self.base_url, timeout=10, verify=True)
                # Additional SSL tests would go here
                # For now, just check if HTTPS is working
                results["ssl_working"] = True
            except requests.exceptions.SSLError as e:
                results["vulnerabilities"].append({
                    "type": "ssl_error",
                    "error": str(e),
                    "severity": "high"
                })
                results["status"] = "vulnerable"
        else:
            results["vulnerabilities"].append({
                "type": "no_https",
                "message": "Application not using HTTPS",
                "severity": "medium"
            })
            results["status"] = "vulnerable"
        
        return results
    
    def test_security_headers(self):
        """Test security headers"""
        logger.info("Testing security headers...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        try:
            response = requests.get(self.base_url, timeout=10)
            headers = response.headers
            
            # Check for important security headers
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
                'X-XSS-Protection': '1; mode=block',
                'Strict-Transport-Security': None,  # Any value is good
                'Content-Security-Policy': None,
                'Referrer-Policy': None
            }
            
            for header, expected_value in security_headers.items():
                if header not in headers:
                    results["vulnerabilities"].append({
                        "type": "missing_security_header",
                        "header": header,
                        "severity": "medium"
                    })
                    results["status"] = "vulnerable"
                elif expected_value and headers[header] not in expected_value:
                    results["vulnerabilities"].append({
                        "type": "weak_security_header",
                        "header": header,
                        "current_value": headers[header],
                        "expected": expected_value,
                        "severity": "low"
                    })
                    
        except requests.RequestException:
            pass
        
        return results
    
    def test_api_security(self):
        """Test API-specific security"""
        logger.info("Testing API security...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Test for API versioning
        try:
            response = requests.get(f"{self.base_url}/api", timeout=10)
            if response.status_code == 200:
                # Check if API exposes version information
                if 'version' in response.text.lower():
                    results["api_version_exposed"] = True
        except requests.RequestException:
            pass
        
        # Test for API documentation exposure
        doc_endpoints = ['/docs', '/swagger', '/api-docs', '/redoc']
        for endpoint in doc_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == 200:
                    results["vulnerabilities"].append({
                        "type": "api_docs_exposed",
                        "endpoint": endpoint,
                        "severity": "low"
                    })
            except requests.RequestException:
                pass
        
        return results
    
    def test_fraud_detection_security(self):
        """Test fraud detection specific security"""
        logger.info("Testing fraud detection security...")
        
        results = {"vulnerabilities": [], "status": "passed"}
        
        # Test for data injection in fraud detection
        malicious_cdr = {
            "caller_number": "1234567890",
            "callee_number": "0987654321",
            "duration": 3600,
            "cost": 999999.99,  # Extremely high cost
            "timestamp": "2099-12-31T23:59:59Z",  # Future timestamp
            "location": "../../../etc/passwd"  # Path injection
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/analyze/cdr",
                json=malicious_cdr,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                # Check if the system properly handles malicious data
                if 'fraud_score' not in result:
                    results["vulnerabilities"].append({
                        "type": "fraud_detection_bypass",
                        "payload": malicious_cdr,
                        "severity": "critical"
                    })
                    results["status"] = "vulnerable"
                    
        except requests.RequestException:
            pass
        
        return results
    
    def save_results(self, results):
        """Save test results to file"""
        results_file = self.results_dir / f"security_test_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"📁 Results saved to {results_file}")
    
    def generate_report(self, results):
        """Generate human-readable security report"""
        report_file = self.results_dir / f"security_report_{self.timestamp}.md"
        
        total_vulnerabilities = sum(
            len(test_result.get('vulnerabilities', [])) 
            for test_result in results['tests'].values() 
            if isinstance(test_result, dict)
        )
        
        vulnerable_tests = sum(
            1 for test_result in results['tests'].values() 
            if isinstance(test_result, dict) and test_result.get('status') == 'vulnerable'
        )
        
        with open(report_file, 'w') as f:
            f.write(f"# FraudGuard 360 Security Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Target:** {self.base_url}\n")
            f.write(f"**Total Tests:** {len(results['tests'])}\n")
            f.write(f"**Vulnerable Tests:** {vulnerable_tests}\n")
            f.write(f"**Total Vulnerabilities:** {total_vulnerabilities}\n\n")
            
            if total_vulnerabilities == 0:
                f.write("## ✅ No Critical Vulnerabilities Found\n\n")
                f.write("The application passed all security tests.\n\n")
            else:
                f.write("## ⚠️ Security Issues Found\n\n")
                
                for test_name, test_result in results['tests'].items():
                    if isinstance(test_result, dict) and test_result.get('vulnerabilities'):
                        f.write(f"### {test_name.replace('_', ' ').title()}\n\n")
                        for vuln in test_result['vulnerabilities']:
                            f.write(f"- **{vuln['type']}** (Severity: {vuln['severity']})\n")
                            for key, value in vuln.items():
                                if key not in ['type', 'severity']:
                                    f.write(f"  - {key}: {value}\n")
                            f.write("\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. Fix all critical and high severity vulnerabilities immediately\n")
            f.write("2. Implement proper input validation and sanitization\n")
            f.write("3. Add security headers to all responses\n")
            f.write("4. Implement rate limiting and CSRF protection\n")
            f.write("5. Regular security testing and code reviews\n")
            f.write("6. Keep all dependencies updated\n")
            f.write("7. Implement proper error handling to avoid information disclosure\n")
        
        logger.info(f"📊 Security report generated: {report_file}")

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FraudGuard 360 Security Testing')
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Base URL to test (default: http://localhost:8000)')
    parser.add_argument('--output', default='security-results',
                       help='Output directory for results (default: security-results)')
    
    args = parser.parse_args()
    
    tester = SecurityTester(base_url=args.url, results_dir=args.output)
    results = tester.run_all_tests()
    
    # Print summary
    total_vulnerabilities = sum(
        len(test_result.get('vulnerabilities', [])) 
        for test_result in results['tests'].values() 
        if isinstance(test_result, dict)
    )
    
    if total_vulnerabilities == 0:
        logger.info("🎉 Security testing completed - No critical vulnerabilities found!")
        sys.exit(0)
    else:
        logger.warning(f"⚠️ Security testing completed - {total_vulnerabilities} vulnerabilities found!")
        sys.exit(1)

if __name__ == "__main__":
    main()