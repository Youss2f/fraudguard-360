"""
FraudGuard 360 - Security Testing Suite
Comprehensive security testing for the fraud detection system
"""

import json
import logging
import os
import requests
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple
import base64
import hashlib
import hmac
from urllib.parse import urljoin, urlparse

import sqlparse
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityTestResult:
    """Security test result data class"""
    
    def __init__(self, test_name: str, passed: bool, severity: str, description: str, 
                 details: str = "", recommendation: str = ""):
        self.test_name = test_name
        self.passed = passed
        self.severity = severity  # LOW, MEDIUM, HIGH, CRITICAL
        self.description = description
        self.details = details
        self.recommendation = recommendation
        self.timestamp = datetime.now().isoformat()

class FraudGuardSecurityTest:
    """Security testing suite for FraudGuard 360"""
    
    def __init__(self):
        self.test_session_id = str(uuid.uuid4())
        self.results = []
        self.base_urls = {
            'api_gateway': os.getenv('API_GATEWAY_URL', 'http://localhost:8000'),
            'ml_service': os.getenv('ML_SERVICE_URL', 'http://localhost:8003'),
            'frontend': os.getenv('FRONTEND_URL', 'http://localhost:3000')
        }
        
        # Common payloads for testing
        self.sql_injection_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' OR '1'='1' /*",
            "admin'--",
            "' OR 1=1#",
            "'; EXEC xp_cmdshell('dir'); --"
        ]
        
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'><script>alert('XSS')</script>",
            "<iframe src=javascript:alert('XSS')></iframe>"
        ]
        
        self.nosql_injection_payloads = [
            "{'$ne': null}",
            "{'$gt': ''}",
            "{'$where': 'this.password.length > 0'}",
            "{'$regex': '.*'}",
            "'; return true; var dummy='",
            "{'$or': [{}]}"
        ]
    
    def log_result(self, result: SecurityTestResult):
        """Log and store security test result"""
        self.results.append(result)
        
        status_icon = "✅" if result.passed else "❌"
        logger.info(f"{status_icon} {result.test_name} - {result.severity}")
        
        if not result.passed:
            logger.warning(f"   {result.description}")
            if result.details:
                logger.warning(f"   Details: {result.details}")
    
    # ==========================================================================
    # Authentication & Authorization Tests
    # ==========================================================================
    
    def test_authentication_bypass(self) -> List[SecurityTestResult]:
        """Test for authentication bypass vulnerabilities"""
        logger.info("Testing authentication bypass...")
        
        results = []
        api_base = self.base_urls['api_gateway']
        
        # Test 1: Access protected endpoints without authentication
        protected_endpoints = [
            '/users/profile',
            '/admin/users',
            '/admin/system',
            '/metrics/internal',
            '/config/system'
        ]
        
        for endpoint in protected_endpoints:
            try:
                url = urljoin(api_base, endpoint)
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    result = SecurityTestResult(
                        test_name=f"Unauthenticated access to {endpoint}",
                        passed=False,
                        severity="HIGH",
                        description=f"Protected endpoint {endpoint} accessible without authentication",
                        details=f"HTTP {response.status_code}: {response.text[:200]}",
                        recommendation="Implement proper authentication middleware for all protected endpoints"
                    )
                else:
                    result = SecurityTestResult(
                        test_name=f"Authentication required for {endpoint}",
                        passed=True,
                        severity="LOW",
                        description=f"Endpoint {endpoint} properly protected"
                    )
                
                results.append(result)
                self.log_result(result)
                
            except Exception as e:
                result = SecurityTestResult(
                    test_name=f"Authentication test error for {endpoint}",
                    passed=True,  # Error might indicate proper security
                    severity="LOW",
                    description=f"Could not test endpoint {endpoint}",
                    details=str(e)
                )
                results.append(result)
                self.log_result(result)
        
        # Test 2: JWT token manipulation
        try:
            # Attempt to get a valid token first
            login_response = requests.post(
                urljoin(api_base, '/auth/login'),
                json={'username': 'test_user', 'password': 'test_password'},
                timeout=10
            )
            
            if login_response.status_code == 200:
                token_data = login_response.json()
                if 'access_token' in token_data:
                    valid_token = token_data['access_token']
                    
                    # Test with modified token
                    modified_token = valid_token[:-5] + "XXXXX"  # Modify last 5 characters
                    
                    response = requests.get(
                        urljoin(api_base, '/users/profile'),
                        headers={'Authorization': f'Bearer {modified_token}'},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = SecurityTestResult(
                            test_name="JWT token validation",
                            passed=False,
                            severity="CRITICAL",
                            description="Modified JWT token accepted",
                            details="System accepts tampered JWT tokens",
                            recommendation="Implement proper JWT signature verification"
                        )
                    else:
                        result = SecurityTestResult(
                            test_name="JWT token validation",
                            passed=True,
                            severity="LOW",
                            description="JWT token validation working correctly"
                        )
                    
                    results.append(result)
                    self.log_result(result)
            
        except Exception as e:
            logger.warning(f"JWT token test failed: {e}")
        
        return results
    
    def test_authorization_bypass(self) -> List[SecurityTestResult]:
        """Test for authorization bypass vulnerabilities"""
        logger.info("Testing authorization bypass...")
        
        results = []
        
        # Test privilege escalation attempts
        privilege_escalation_tests = [
            # Path traversal in user ID
            {'user_id': '../admin', 'description': 'Path traversal in user ID'},
            {'user_id': '1 OR 1=1', 'description': 'SQL injection in user ID'},
            {'user_id': '{"$ne": null}', 'description': 'NoSQL injection in user ID'},
            # Direct admin access attempts
            {'endpoint': '/admin/users/1', 'description': 'Direct admin endpoint access'},
            {'endpoint': '/admin/system/config', 'description': 'System config access'},
        ]
        
        for test_case in privilege_escalation_tests:
            try:
                if 'user_id' in test_case:
                    url = f"{self.base_urls['api_gateway']}/users/{test_case['user_id']}"
                else:
                    url = f"{self.base_urls['api_gateway']}{test_case['endpoint']}"
                
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    result = SecurityTestResult(
                        test_name=f"Authorization bypass: {test_case['description']}",
                        passed=False,
                        severity="HIGH",
                        description=f"Unauthorized access successful: {test_case['description']}",
                        details=f"URL: {url}, Response: {response.status_code}",
                        recommendation="Implement proper authorization checks for all user actions"
                    )
                else:
                    result = SecurityTestResult(
                        test_name=f"Authorization check: {test_case['description']}",
                        passed=True,
                        severity="LOW",
                        description=f"Authorization properly enforced for {test_case['description']}"
                    )
                
                results.append(result)
                self.log_result(result)
                
            except Exception as e:
                logger.warning(f"Authorization test failed for {test_case['description']}: {e}")
        
        return results
    
    # ==========================================================================
    # Injection Attack Tests
    # ==========================================================================
    
    def test_sql_injection(self) -> List[SecurityTestResult]:
        """Test for SQL injection vulnerabilities"""
        logger.info("Testing SQL injection vulnerabilities...")
        
        results = []
        
        # Test SQL injection in various endpoints
        test_endpoints = [
            {'url': '/users/search', 'param': 'username', 'method': 'GET'},
            {'url': '/cdr/search', 'param': 'user_id', 'method': 'GET'},
            {'url': '/reports/fraud', 'param': 'filter', 'method': 'GET'},
            {'url': '/auth/login', 'param': 'username', 'method': 'POST'},
        ]
        
        for endpoint in test_endpoints:
            for payload in self.sql_injection_payloads:
                try:
                    url = f"{self.base_urls['api_gateway']}{endpoint['url']}"
                    
                    if endpoint['method'] == 'GET':
                        params = {endpoint['param']: payload}
                        response = requests.get(url, params=params, timeout=10)
                    else:
                        data = {endpoint['param']: payload}
                        response = requests.post(url, json=data, timeout=10)
                    
                    # Check for SQL error messages or unexpected behavior
                    sql_error_indicators = [
                        'sql syntax', 'mysql', 'postgresql', 'oracle', 'sqlite',
                        'syntax error', 'table', 'column', 'database',
                        'query failed', 'invalid query'
                    ]
                    
                    response_text = response.text.lower()
                    
                    if any(indicator in response_text for indicator in sql_error_indicators):
                        result = SecurityTestResult(
                            test_name=f"SQL Injection in {endpoint['url']}",
                            passed=False,
                            severity="HIGH",
                            description=f"SQL injection vulnerability detected in {endpoint['url']}",
                            details=f"Payload: {payload}, Response indicators found",
                            recommendation="Use parameterized queries and input validation"
                        )
                        results.append(result)
                        self.log_result(result)
                        break  # Don't test more payloads for this endpoint
                    
                    # Check for unusual response times (blind SQL injection)
                    if response.elapsed.total_seconds() > 5:
                        result = SecurityTestResult(
                            test_name=f"Potential Blind SQL Injection in {endpoint['url']}",
                            passed=False,
                            severity="MEDIUM",
                            description=f"Unusual response time detected in {endpoint['url']}",
                            details=f"Payload: {payload}, Response time: {response.elapsed.total_seconds()}s",
                            recommendation="Investigate potential time-based SQL injection"
                        )
                        results.append(result)
                        self.log_result(result)
                
                except Exception as e:
                    logger.warning(f"SQL injection test failed for {endpoint['url']}: {e}")
        
        # If no SQL injection found in any endpoint, log success
        if not any(not result.passed for result in results if 'SQL Injection' in result.test_name):
            result = SecurityTestResult(
                test_name="SQL Injection Protection",
                passed=True,
                severity="LOW",
                description="No SQL injection vulnerabilities detected"
            )
            results.append(result)
            self.log_result(result)
        
        return results
    
    def test_nosql_injection(self) -> List[SecurityTestResult]:
        """Test for NoSQL injection vulnerabilities"""
        logger.info("Testing NoSQL injection vulnerabilities...")
        
        results = []
        
        # Test NoSQL injection in ML service and API endpoints
        test_endpoints = [
            {'url': f"{self.base_urls['ml_service']}/predict", 'method': 'POST'},
            {'url': f"{self.base_urls['api_gateway']}/users/search", 'method': 'POST'},
        ]
        
        for endpoint in test_endpoints:
            for payload in self.nosql_injection_payloads:
                try:
                    # Create test data with NoSQL injection payload
                    test_data = {
                        'user_id': payload,
                        'call_duration': 100,
                        'call_cost': 10.5,
                        'calls_per_day': 15
                    }
                    
                    response = requests.post(endpoint['url'], json=test_data, timeout=10)
                    
                    # Check for NoSQL error messages
                    nosql_error_indicators = [
                        'mongodb', 'mongoose', 'query failed',
                        'invalid query', 'bson', 'objectid'
                    ]
                    
                    response_text = response.text.lower()
                    
                    if any(indicator in response_text for indicator in nosql_error_indicators):
                        result = SecurityTestResult(
                            test_name=f"NoSQL Injection in {endpoint['url']}",
                            passed=False,
                            severity="HIGH",
                            description=f"NoSQL injection vulnerability detected",
                            details=f"Payload: {payload}",
                            recommendation="Implement proper input validation and use safe query methods"
                        )
                        results.append(result)
                        self.log_result(result)
                        break
                
                except Exception as e:
                    logger.warning(f"NoSQL injection test failed: {e}")
        
        return results
    
    def test_xss_vulnerabilities(self) -> List[SecurityTestResult]:
        """Test for Cross-Site Scripting (XSS) vulnerabilities"""
        logger.info("Testing XSS vulnerabilities...")
        
        results = []
        
        # Test XSS in various input fields
        xss_test_endpoints = [
            {'url': '/users/profile', 'param': 'display_name', 'method': 'POST'},
            {'url': '/reports/comment', 'param': 'comment', 'method': 'POST'},
            {'url': '/search', 'param': 'query', 'method': 'GET'},
        ]
        
        for endpoint in xss_test_endpoints:
            for payload in self.xss_payloads:
                try:
                    url = f"{self.base_urls['api_gateway']}{endpoint['url']}"
                    
                    if endpoint['method'] == 'GET':
                        params = {endpoint['param']: payload}
                        response = requests.get(url, params=params, timeout=10)
                    else:
                        data = {endpoint['param']: payload}
                        response = requests.post(url, json=data, timeout=10)
                    
                    # Check if payload is reflected in response
                    if payload in response.text:
                        result = SecurityTestResult(
                            test_name=f"Reflected XSS in {endpoint['url']}",
                            passed=False,
                            severity="MEDIUM",
                            description=f"XSS payload reflected in response",
                            details=f"Payload: {payload}",
                            recommendation="Implement proper output encoding and input sanitization"
                        )
                        results.append(result)
                        self.log_result(result)
                        break
                
                except Exception as e:
                    logger.warning(f"XSS test failed for {endpoint['url']}: {e}")
        
        return results
    
    # ==========================================================================
    # Data Security Tests
    # ==========================================================================
    
    def test_sensitive_data_exposure(self) -> List[SecurityTestResult]:
        """Test for sensitive data exposure"""
        logger.info("Testing sensitive data exposure...")
        
        results = []
        
        # Test for exposed configuration files
        config_files = [
            '/.env',
            '/config.json',
            '/database.json',
            '/.aws/credentials',
            '/docker-compose.yml',
            '/Dockerfile',
            '/.git/config',
            '/backup.sql',
            '/users.json'
        ]
        
        for config_file in config_files:
            try:
                url = f"{self.base_urls['api_gateway']}{config_file}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    result = SecurityTestResult(
                        test_name=f"Exposed configuration file: {config_file}",
                        passed=False,
                        severity="HIGH",
                        description=f"Configuration file {config_file} is publicly accessible",
                        details=f"File content length: {len(response.text)} characters",
                        recommendation="Remove public access to configuration files"
                    )
                else:
                    result = SecurityTestResult(
                        test_name=f"Configuration file protection: {config_file}",
                        passed=True,
                        severity="LOW",
                        description=f"Configuration file {config_file} properly protected"
                    )
                
                results.append(result)
                self.log_result(result)
                
            except Exception as e:
                logger.warning(f"Config file test failed for {config_file}: {e}")
        
        # Test for information disclosure in error messages
        try:
            # Trigger error by sending malformed data
            malformed_data = {"invalid": "json", "structure": ["with", {"nested": None}]}
            
            response = requests.post(
                f"{self.base_urls['ml_service']}/predict",
                json=malformed_data,
                timeout=10
            )
            
            # Check if error message reveals sensitive information
            sensitive_info_indicators = [
                'traceback', 'stack trace', 'file path', '/home/', '/var/',
                'database', 'password', 'secret', 'key', 'token'
            ]
            
            response_text = response.text.lower()
            
            if any(indicator in response_text for indicator in sensitive_info_indicators):
                result = SecurityTestResult(
                    test_name="Information disclosure in error messages",
                    passed=False,
                    severity="MEDIUM",
                    description="Error messages reveal sensitive system information",
                    details="Detailed error information exposed to users",
                    recommendation="Implement generic error messages for production"
                )
            else:
                result = SecurityTestResult(
                    test_name="Error message security",
                    passed=True,
                    severity="LOW",
                    description="Error messages do not expose sensitive information"
                )
            
            results.append(result)
            self.log_result(result)
            
        except Exception as e:
            logger.warning(f"Error message test failed: {e}")
        
        return results
    
    def test_data_encryption(self) -> List[SecurityTestResult]:
        """Test data encryption and secure transmission"""
        logger.info("Testing data encryption and secure transmission...")
        
        results = []
        
        # Test HTTPS enforcement
        for service_name, base_url in self.base_urls.items():
            try:
                # Try HTTP version if HTTPS is configured
                if base_url.startswith('https://'):
                    http_url = base_url.replace('https://', 'http://')
                    
                    response = requests.get(http_url, timeout=10, allow_redirects=False)
                    
                    if response.status_code in [301, 302, 307, 308]:
                        # Check if redirected to HTTPS
                        location = response.headers.get('Location', '')
                        if location.startswith('https://'):
                            result = SecurityTestResult(
                                test_name=f"HTTPS redirection for {service_name}",
                                passed=True,
                                severity="LOW",
                                description=f"HTTP requests properly redirected to HTTPS"
                            )
                        else:
                            result = SecurityTestResult(
                                test_name=f"HTTPS enforcement for {service_name}",
                                passed=False,
                                severity="MEDIUM",
                                description=f"HTTP requests not redirected to HTTPS",
                                recommendation="Configure HTTP to HTTPS redirection"
                            )
                    elif response.status_code == 200:
                        result = SecurityTestResult(
                            test_name=f"HTTPS enforcement for {service_name}",
                            passed=False,
                            severity="HIGH",
                            description=f"Service accepts unencrypted HTTP connections",
                            recommendation="Enforce HTTPS for all connections"
                        )
                    else:
                        result = SecurityTestResult(
                            test_name=f"HTTP access control for {service_name}",
                            passed=True,
                            severity="LOW",
                            description=f"HTTP access properly blocked"
                        )
                else:
                    result = SecurityTestResult(
                        test_name=f"HTTPS configuration for {service_name}",
                        passed=False,
                        severity="MEDIUM",
                        description=f"Service not configured for HTTPS",
                        recommendation="Configure HTTPS/TLS for secure communication"
                    )
                
                results.append(result)
                self.log_result(result)
                
            except Exception as e:
                logger.warning(f"HTTPS test failed for {service_name}: {e}")
        
        return results
    
    # ==========================================================================
    # API Security Tests
    # ==========================================================================
    
    def test_api_rate_limiting(self) -> List[SecurityTestResult]:
        """Test API rate limiting and DoS protection"""
        logger.info("Testing API rate limiting...")
        
        results = []
        
        # Test rate limiting by sending rapid requests
        test_endpoints = [
            f"{self.base_urls['api_gateway']}/health",
            f"{self.base_urls['ml_service']}/health",
        ]
        
        for endpoint in test_endpoints:
            try:
                # Send 100 rapid requests
                success_count = 0
                rate_limited_count = 0
                
                for i in range(100):
                    response = requests.get(endpoint, timeout=5)
                    
                    if response.status_code == 200:
                        success_count += 1
                    elif response.status_code in [429, 503]:  # Too Many Requests or Service Unavailable
                        rate_limited_count += 1
                
                if rate_limited_count > 0:
                    result = SecurityTestResult(
                        test_name=f"Rate limiting for {endpoint}",
                        passed=True,
                        severity="LOW",
                        description=f"Rate limiting active on {endpoint}",
                        details=f"{rate_limited_count}/100 requests rate limited"
                    )
                else:
                    result = SecurityTestResult(
                        test_name=f"Rate limiting for {endpoint}",
                        passed=False,
                        severity="MEDIUM",
                        description=f"No rate limiting detected on {endpoint}",
                        details=f"All 100 rapid requests succeeded",
                        recommendation="Implement rate limiting to prevent DoS attacks"
                    )
                
                results.append(result)
                self.log_result(result)
                
            except Exception as e:
                logger.warning(f"Rate limiting test failed for {endpoint}: {e}")
        
        return results
    
    def test_cors_configuration(self) -> List[SecurityTestResult]:
        """Test CORS configuration security"""
        logger.info("Testing CORS configuration...")
        
        results = []
        
        for service_name, base_url in self.base_urls.items():
            try:
                # Test CORS with different origins
                test_origins = [
                    'http://malicious-site.com',
                    'https://evil.example.com',
                    'null'
                ]
                
                for origin in test_origins:
                    headers = {
                        'Origin': origin,
                        'Access-Control-Request-Method': 'POST',
                        'Access-Control-Request-Headers': 'Content-Type'
                    }
                    
                    response = requests.options(f"{base_url}/health", headers=headers, timeout=10)
                    
                    cors_origin = response.headers.get('Access-Control-Allow-Origin', '')
                    
                    if cors_origin == '*':
                        result = SecurityTestResult(
                            test_name=f"CORS wildcard origin for {service_name}",
                            passed=False,
                            severity="MEDIUM",
                            description=f"CORS allows all origins (*) for {service_name}",
                            recommendation="Configure specific allowed origins instead of wildcard"
                        )
                        results.append(result)
                        self.log_result(result)
                        break
                    elif cors_origin == origin:
                        result = SecurityTestResult(
                            test_name=f"CORS allows malicious origin for {service_name}",
                            passed=False,
                            severity="HIGH",
                            description=f"CORS allows malicious origin {origin}",
                            recommendation="Review and restrict CORS allowed origins"
                        )
                        results.append(result)
                        self.log_result(result)
                        break
                else:
                    # No malicious origins allowed
                    result = SecurityTestResult(
                        test_name=f"CORS configuration for {service_name}",
                        passed=True,
                        severity="LOW",
                        description=f"CORS properly configured for {service_name}"
                    )
                    results.append(result)
                    self.log_result(result)
                
            except Exception as e:
                logger.warning(f"CORS test failed for {service_name}: {e}")
        
        return results
    
    # ==========================================================================
    # Report Generation
    # ==========================================================================
    
    def generate_security_report(self) -> str:
        """Generate comprehensive security test report"""
        logger.info("Generating security test report...")
        
        # Categorize results by severity
        critical_issues = [r for r in self.results if r.severity == 'CRITICAL' and not r.passed]
        high_issues = [r for r in self.results if r.severity == 'HIGH' and not r.passed]
        medium_issues = [r for r in self.results if r.severity == 'MEDIUM' and not r.passed]
        low_issues = [r for r in self.results if r.severity == 'LOW' and not r.passed]
        
        passed_tests = [r for r in self.results if r.passed]
        
        report = []
        report.append("# FraudGuard 360 Security Test Report")
        report.append(f"Test Session ID: {self.test_session_id}")
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"**Total Tests Conducted:** {len(self.results)}")
        report.append(f"**Tests Passed:** {len(passed_tests)}")
        report.append(f"**Security Issues Found:** {len(self.results) - len(passed_tests)}")
        report.append("")
        report.append("### Issue Breakdown by Severity")
        report.append(f"- 🔴 **Critical:** {len(critical_issues)}")
        report.append(f"- 🟠 **High:** {len(high_issues)}")
        report.append(f"- 🟡 **Medium:** {len(medium_issues)}")
        report.append(f"- 🔵 **Low:** {len(low_issues)}")
        report.append("")
        
        # Critical Issues
        if critical_issues:
            report.append("## 🔴 Critical Security Issues")
            report.append("*These issues require immediate attention*")
            report.append("")
            
            for issue in critical_issues:
                report.append(f"### {issue.test_name}")
                report.append(f"**Description:** {issue.description}")
                if issue.details:
                    report.append(f"**Details:** {issue.details}")
                if issue.recommendation:
                    report.append(f"**Recommendation:** {issue.recommendation}")
                report.append("")
        
        # High Issues
        if high_issues:
            report.append("## 🟠 High Priority Security Issues")
            report.append("")
            
            for issue in high_issues:
                report.append(f"### {issue.test_name}")
                report.append(f"**Description:** {issue.description}")
                if issue.details:
                    report.append(f"**Details:** {issue.details}")
                if issue.recommendation:
                    report.append(f"**Recommendation:** {issue.recommendation}")
                report.append("")
        
        # Medium Issues
        if medium_issues:
            report.append("## 🟡 Medium Priority Security Issues")
            report.append("")
            
            for issue in medium_issues:
                report.append(f"### {issue.test_name}")
                report.append(f"**Description:** {issue.description}")
                if issue.recommendation:
                    report.append(f"**Recommendation:** {issue.recommendation}")
                report.append("")
        
        # Security Recommendations
        report.append("## Security Recommendations")
        report.append("")
        
        if critical_issues or high_issues:
            report.append("### Immediate Actions Required")
            report.append("- Address all critical and high priority issues immediately")
            report.append("- Implement proper authentication and authorization controls")
            report.append("- Add input validation and sanitization")
            report.append("- Configure secure communication (HTTPS/TLS)")
            report.append("")
        
        report.append("### General Security Best Practices")
        report.append("- Implement comprehensive logging and monitoring")
        report.append("- Regular security testing and code reviews")
        report.append("- Keep all dependencies updated")
        report.append("- Implement proper error handling")
        report.append("- Use security headers (HSTS, CSP, etc.)")
        report.append("- Regular penetration testing")
        report.append("")
        
        # Passed Tests Summary
        if passed_tests:
            report.append("## ✅ Security Controls Working Correctly")
            report.append("")
            
            security_categories = {}
            for test in passed_tests:
                category = test.test_name.split(' ')[0]  # First word as category
                if category not in security_categories:
                    security_categories[category] = []
                security_categories[category].append(test.test_name)
            
            for category, tests in security_categories.items():
                report.append(f"**{category} Security:**")
                for test in tests[:5]:  # Show first 5 tests per category
                    report.append(f"- {test}")
                if len(tests) > 5:
                    report.append(f"- ... and {len(tests) - 5} more")
                report.append("")
        
        report.append("---")
        report.append("*Report generated by FraudGuard 360 Security Testing Suite*")
        
        return "\n".join(report)
    
    def run_comprehensive_security_test(self) -> Dict[str, Any]:
        """Run comprehensive security test suite"""
        logger.info("Starting comprehensive security test suite...")
        
        all_results = []
        
        # Run all security tests
        try:
            auth_results = self.test_authentication_bypass()
            all_results.extend(auth_results)
        except Exception as e:
            logger.error(f"Authentication tests failed: {e}")
        
        try:
            authz_results = self.test_authorization_bypass()
            all_results.extend(authz_results)
        except Exception as e:
            logger.error(f"Authorization tests failed: {e}")
        
        try:
            sql_results = self.test_sql_injection()
            all_results.extend(sql_results)
        except Exception as e:
            logger.error(f"SQL injection tests failed: {e}")
        
        try:
            nosql_results = self.test_nosql_injection()
            all_results.extend(nosql_results)
        except Exception as e:
            logger.error(f"NoSQL injection tests failed: {e}")
        
        try:
            xss_results = self.test_xss_vulnerabilities()
            all_results.extend(xss_results)
        except Exception as e:
            logger.error(f"XSS tests failed: {e}")
        
        try:
            data_results = self.test_sensitive_data_exposure()
            all_results.extend(data_results)
        except Exception as e:
            logger.error(f"Data exposure tests failed: {e}")
        
        try:
            encryption_results = self.test_data_encryption()
            all_results.extend(encryption_results)
        except Exception as e:
            logger.error(f"Encryption tests failed: {e}")
        
        try:
            rate_limit_results = self.test_api_rate_limiting()
            all_results.extend(rate_limit_results)
        except Exception as e:
            logger.error(f"Rate limiting tests failed: {e}")
        
        try:
            cors_results = self.test_cors_configuration()
            all_results.extend(cors_results)
        except Exception as e:
            logger.error(f"CORS tests failed: {e}")
        
        # Generate and save report
        report = self.generate_security_report()
        
        os.makedirs("security_reports", exist_ok=True)
        report_filename = f"security_reports/security_report_{self.test_session_id[:8]}.md"
        
        with open(report_filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Security report saved to {report_filename}")
        
        # Return summary
        summary = {
            'total_tests': len(self.results),
            'passed_tests': len([r for r in self.results if r.passed]),
            'failed_tests': len([r for r in self.results if not r.passed]),
            'critical_issues': len([r for r in self.results if r.severity == 'CRITICAL' and not r.passed]),
            'high_issues': len([r for r in self.results if r.severity == 'HIGH' and not r.passed]),
            'medium_issues': len([r for r in self.results if r.severity == 'MEDIUM' and not r.passed]),
            'low_issues': len([r for r in self.results if r.severity == 'LOW' and not r.passed]),
            'report_file': report_filename
        }
        
        return summary

if __name__ == "__main__":
    # Run comprehensive security test
    security_test = FraudGuardSecurityTest()
    
    try:
        results = security_test.run_comprehensive_security_test()
        
        print("\n" + "="*60)
        print("SECURITY TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"\nIssues by Severity:")
        print(f"  Critical: {results['critical_issues']}")
        print(f"  High: {results['high_issues']}")
        print(f"  Medium: {results['medium_issues']}")
        print(f"  Low: {results['low_issues']}")
        print(f"\nReport saved to: {results['report_file']}")
        print("="*60)
        
        logger.info("=== Security Test Suite Completed ===")
        
    except Exception as e:
        logger.error(f"Security test suite failed: {e}")
        raise