"""
FraudGuard 360 - Comprehensive Test Runner
Orchestrates all testing suites: unit, integration, performance, and security
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_runner.log')
    ]
)
logger = logging.getLogger(__name__)

class TestSuite:
    """Individual test suite configuration"""
    
    def __init__(self, name: str, path: str, description: str, 
                 prerequisites: List[str] = None, timeout: int = 300):
        self.name = name
        self.path = path
        self.description = description
        self.prerequisites = prerequisites or []
        self.timeout = timeout
        self.results = {}
        self.status = "not_run"  # not_run, running, passed, failed, skipped

class FraudGuardTestRunner:
    """Main test runner for FraudGuard 360 system"""
    
    def __init__(self):
        self.test_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.workspace_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Define all test suites
        self.test_suites = {
            "unit_api_gateway": TestSuite(
                name="API Gateway Unit Tests",
                path="tests/test_api_gateway.py",
                description="Unit tests for API Gateway service",
                timeout=120
            ),
            "unit_ml_service": TestSuite(
                name="ML Service Unit Tests", 
                path="ml-service/tests/test_ml_inference.py",
                description="Unit tests for ML inference service",
                timeout=180
            ),
            "unit_frontend": TestSuite(
                name="Frontend Unit Tests",
                path="frontend/src",
                description="React component unit tests",
                timeout=120
            ),
            "integration_pipeline": TestSuite(
                name="Integration Tests",
                path="tests/integration/test_fraud_detection_pipeline.py",
                description="End-to-end pipeline integration tests",
                prerequisites=["unit_api_gateway", "unit_ml_service"],
                timeout=600
            ),
            "performance_load": TestSuite(
                name="Performance Tests",
                path="tests/performance/test_load_performance.py", 
                description="Load testing and performance validation",
                prerequisites=["integration_pipeline"],
                timeout=900
            ),
            "security_vulnerabilities": TestSuite(
                name="Security Tests",
                path="tests/security/test_security_vulnerabilities.py",
                description="Security vulnerability testing",
                timeout=600
            )
        }
        
        # Service health check endpoints
        self.service_endpoints = {
            'api_gateway': os.getenv('API_GATEWAY_URL', 'http://localhost:8000'),
            'ml_service': os.getenv('ML_SERVICE_URL', 'http://localhost:8003'),
            'frontend': os.getenv('FRONTEND_URL', 'http://localhost:3000')
        }
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check system prerequisites for running tests"""
        logger.info("Checking system prerequisites...")
        
        prerequisites = {
            'python': False,
            'node': False,
            'docker': False,
            'services_running': False,
            'test_data': False
        }
        
        # Check Python
        try:
            result = subprocess.run(['python', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                prerequisites['python'] = True
                logger.info(f"✓ Python: {result.stdout.strip()}")
        except Exception as e:
            logger.warning(f"❌ Python check failed: {e}")
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                prerequisites['node'] = True
                logger.info(f"✓ Node.js: {result.stdout.strip()}")
        except Exception as e:
            logger.warning(f"❌ Node.js check failed: {e}")
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                prerequisites['docker'] = True
                logger.info(f"✓ Docker: {result.stdout.strip()}")
        except Exception as e:
            logger.warning(f"❌ Docker check failed: {e}")
        
        # Check if services are running
        prerequisites['services_running'] = self.check_service_health()
        
        # Check test data availability
        test_data_files = [
            'ml-service/data/synthetic_fraud_data.csv',
            'tests/test_data.json'
        ]
        
        prerequisites['test_data'] = all(
            (self.workspace_root / file).exists() 
            for file in test_data_files
        )
        
        return prerequisites
    
    def check_service_health(self) -> bool:
        """Check if all required services are running"""
        import requests
        
        healthy_services = 0
        total_services = len(self.service_endpoints)
        
        for service_name, endpoint in self.service_endpoints.items():
            try:
                response = requests.get(f"{endpoint}/health", timeout=5)
                if response.status_code == 200:
                    healthy_services += 1
                    logger.info(f"✓ {service_name} is healthy")
                else:
                    logger.warning(f"❌ {service_name} health check failed: {response.status_code}")
            except Exception as e:
                logger.warning(f"❌ {service_name} is not reachable: {e}")
        
        services_healthy = healthy_services == total_services
        logger.info(f"Services health: {healthy_services}/{total_services} healthy")
        return services_healthy
    
    def setup_test_environment(self):
        """Setup test environment and dependencies"""
        logger.info("Setting up test environment...")
        
        # Create test output directories
        test_dirs = [
            'test_reports',
            'performance_reports', 
            'security_reports',
            'coverage_reports'
        ]
        
        for dir_name in test_dirs:
            dir_path = self.workspace_root / dir_name
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Install Python test dependencies
        try:
            requirements_file = self.workspace_root / 'tests' / 'requirements.txt'
            if requirements_file.exists():
                subprocess.run([
                    'pip', 'install', '-r', str(requirements_file)
                ], check=True)
                logger.info("✓ Python test dependencies installed")
        except Exception as e:
            logger.warning(f"Failed to install Python test dependencies: {e}")
        
        # Install Node.js test dependencies for frontend
        try:
            frontend_dir = self.workspace_root / 'frontend'
            if (frontend_dir / 'package.json').exists():
                subprocess.run([
                    'npm', 'install'
                ], cwd=frontend_dir, check=True)
                logger.info("✓ Node.js test dependencies installed")
        except Exception as e:
            logger.warning(f"Failed to install Node.js test dependencies: {e}")
    
    def run_unit_tests(self, suite_name: str) -> Dict[str, Any]:
        """Run unit tests for a specific suite"""
        suite = self.test_suites[suite_name]
        logger.info(f"Running {suite.name}...")
        
        suite.status = "running"
        start_time = time.time()
        
        try:
            if suite_name == "unit_frontend":
                # Run React tests
                result = subprocess.run([
                    'npm', 'test', '--', '--watchAll=false', '--coverage'
                ], 
                cwd=self.workspace_root / 'frontend',
                capture_output=True, 
                text=True,
                timeout=suite.timeout
                )
            else:
                # Run Python tests with pytest
                test_file = self.workspace_root / suite.path
                result = subprocess.run([
                    'python', '-m', 'pytest', str(test_file), 
                    '-v', '--tb=short', '--json-report',
                    f'--json-report-file=test_reports/{suite_name}_results.json'
                ], 
                capture_output=True, 
                text=True,
                timeout=suite.timeout
                )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.returncode == 0:
                suite.status = "passed"
                logger.info(f"✅ {suite.name} passed ({execution_time:.2f}s)")
            else:
                suite.status = "failed"
                logger.error(f"❌ {suite.name} failed ({execution_time:.2f}s)")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
            
            return {
                'status': suite.status,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            suite.status = "failed"
            logger.error(f"❌ {suite.name} timed out after {suite.timeout}s")
            return {
                'status': 'failed',
                'execution_time': suite.timeout,
                'error': 'Test timed out'
            }
        except Exception as e:
            suite.status = "failed"
            logger.error(f"❌ {suite.name} failed with exception: {e}")
            return {
                'status': 'failed',
                'execution_time': time.time() - start_time,
                'error': str(e)
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        suite = self.test_suites["integration_pipeline"]
        logger.info(f"Running {suite.name}...")
        
        suite.status = "running"
        start_time = time.time()
        
        try:
            # Check if services are healthy before integration tests
            if not self.check_service_health():
                logger.warning("Some services are not healthy, but proceeding with integration tests...")
            
            # Run integration tests
            test_file = self.workspace_root / suite.path
            result = subprocess.run([
                'python', str(test_file)
            ], 
            capture_output=True, 
            text=True,
            timeout=suite.timeout
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.returncode == 0:
                suite.status = "passed"
                logger.info(f"✅ {suite.name} passed ({execution_time:.2f}s)")
            else:
                suite.status = "failed"
                logger.error(f"❌ {suite.name} failed ({execution_time:.2f}s)")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
            
            return {
                'status': suite.status,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except Exception as e:
            suite.status = "failed"
            logger.error(f"❌ {suite.name} failed with exception: {e}")
            return {
                'status': 'failed',
                'execution_time': time.time() - start_time,
                'error': str(e)
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        suite = self.test_suites["performance_load"]
        logger.info(f"Running {suite.name}...")
        
        suite.status = "running" 
        start_time = time.time()
        
        try:
            test_file = self.workspace_root / suite.path
            result = subprocess.run([
                'python', str(test_file)
            ], 
            capture_output=True, 
            text=True,
            timeout=suite.timeout
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.returncode == 0:
                suite.status = "passed"
                logger.info(f"✅ {suite.name} passed ({execution_time:.2f}s)")
            else:
                suite.status = "failed"
                logger.error(f"❌ {suite.name} failed ({execution_time:.2f}s)")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
            
            return {
                'status': suite.status,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except Exception as e:
            suite.status = "failed"
            logger.error(f"❌ {suite.name} failed with exception: {e}")
            return {
                'status': 'failed',
                'execution_time': time.time() - start_time,
                'error': str(e)
            }
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        suite = self.test_suites["security_vulnerabilities"]
        logger.info(f"Running {suite.name}...")
        
        suite.status = "running"
        start_time = time.time()
        
        try:
            test_file = self.workspace_root / suite.path
            result = subprocess.run([
                'python', str(test_file)
            ], 
            capture_output=True, 
            text=True,
            timeout=suite.timeout
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            if result.returncode == 0:
                suite.status = "passed"
                logger.info(f"✅ {suite.name} passed ({execution_time:.2f}s)")
            else:
                suite.status = "failed"
                logger.error(f"❌ {suite.name} failed ({execution_time:.2f}s)")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
            
            return {
                'status': suite.status,
                'execution_time': execution_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except Exception as e:
            suite.status = "failed"
            logger.error(f"❌ {suite.name} failed with exception: {e}")
            return {
                'status': 'failed',
                'execution_time': time.time() - start_time,
                'error': str(e)
            }
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive test report"""
        logger.info("Generating comprehensive test report...")
        
        total_execution_time = self.end_time - self.start_time if self.end_time else 0
        
        # Count results by status
        passed_suites = sum(1 for suite in self.test_suites.values() if suite.status == "passed")
        failed_suites = sum(1 for suite in self.test_suites.values() if suite.status == "failed")
        skipped_suites = sum(1 for suite in self.test_suites.values() if suite.status == "skipped")
        total_suites = len(self.test_suites)
        
        report = []
        report.append("# FraudGuard 360 - Comprehensive Test Report")
        report.append(f"Test Session ID: {self.test_session_id}")
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Execution Time: {total_execution_time:.2f} seconds")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"**Total Test Suites:** {total_suites}")
        report.append(f"**Passed:** {passed_suites} ✅")
        report.append(f"**Failed:** {failed_suites} ❌")
        report.append(f"**Skipped:** {skipped_suites} ⏭️")
        report.append(f"**Success Rate:** {(passed_suites / total_suites * 100):.1f}%")
        report.append("")
        
        # Test Suite Results
        report.append("## Test Suite Results")
        report.append("")
        
        for suite_name, suite in self.test_suites.items():
            status_icon = {
                "passed": "✅",
                "failed": "❌", 
                "skipped": "⏭️",
                "not_run": "⏸️"
            }.get(suite.status, "❓")
            
            report.append(f"### {status_icon} {suite.name}")
            report.append(f"**Status:** {suite.status.upper()}")
            report.append(f"**Description:** {suite.description}")
            
            if suite_name in self.test_results:
                result = self.test_results[suite_name]
                if 'execution_time' in result:
                    report.append(f"**Execution Time:** {result['execution_time']:.2f}s")
                
                if suite.status == "failed" and 'error' in result:
                    report.append(f"**Error:** {result['error']}")
                elif suite.status == "failed" and 'stderr' in result:
                    report.append(f"**Error Output:** ```{result['stderr'][:200]}...```")
            
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Test Results")
        report.append("")
        
        # Unit Tests Summary
        unit_test_suites = [name for name in self.test_suites.keys() if name.startswith('unit_')]
        if unit_test_suites:
            report.append("### Unit Tests")
            unit_passed = sum(1 for name in unit_test_suites if self.test_suites[name].status == "passed")
            unit_total = len(unit_test_suites)
            report.append(f"- **Unit Test Success Rate:** {(unit_passed / unit_total * 100):.1f}% ({unit_passed}/{unit_total})")
            report.append("")
        
        # Integration Tests
        if "integration_pipeline" in self.test_results:
            report.append("### Integration Tests")
            integration_result = self.test_results["integration_pipeline"]
            report.append(f"- **Status:** {integration_result.get('status', 'unknown').upper()}")
            report.append(f"- **Execution Time:** {integration_result.get('execution_time', 0):.2f}s")
            report.append("")
        
        # Performance Tests
        if "performance_load" in self.test_results:
            report.append("### Performance Tests")
            perf_result = self.test_results["performance_load"]
            report.append(f"- **Status:** {perf_result.get('status', 'unknown').upper()}")
            report.append(f"- **Execution Time:** {perf_result.get('execution_time', 0):.2f}s")
            
            # Check for performance report
            perf_reports = list(Path('performance_reports').glob('*.md'))
            if perf_reports:
                report.append(f"- **Detailed Report:** {perf_reports[0].name}")
            report.append("")
        
        # Security Tests
        if "security_vulnerabilities" in self.test_results:
            report.append("### Security Tests")
            security_result = self.test_results["security_vulnerabilities"]
            report.append(f"- **Status:** {security_result.get('status', 'unknown').upper()}")
            report.append(f"- **Execution Time:** {security_result.get('execution_time', 0):.2f}s")
            
            # Check for security report
            security_reports = list(Path('security_reports').glob('*.md'))
            if security_reports:
                report.append(f"- **Detailed Report:** {security_reports[0].name}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        if failed_suites > 0:
            report.append("### Immediate Actions Required")
            report.append("- Address all failed test suites before deployment")
            report.append("- Review error logs and fix underlying issues")
            report.append("- Re-run tests after fixes are implemented")
            report.append("")
        
        report.append("### Continuous Improvement")
        report.append("- Implement automated testing in CI/CD pipeline")
        report.append("- Regular performance monitoring and testing")
        report.append("- Security testing should be part of regular development cycle")
        report.append("- Consider implementing chaos engineering for resilience testing")
        report.append("")
        
        # Test Coverage and Quality Metrics
        report.append("## Quality Metrics")
        report.append("")
        report.append("### Test Coverage")
        report.append("| Component | Unit Tests | Integration | Performance | Security |")
        report.append("|-----------|------------|-------------|-------------|----------|")
        
        components = ['API Gateway', 'ML Service', 'Frontend', 'End-to-End Pipeline']
        for component in components:
            unit_status = "✅" if any(self.test_suites[k].status == "passed" for k in self.test_suites if k.startswith('unit_')) else "❌"
            integration_status = "✅" if self.test_suites.get("integration_pipeline", {}).status == "passed" else "❌"
            performance_status = "✅" if self.test_suites.get("performance_load", {}).status == "passed" else "❌"
            security_status = "✅" if self.test_suites.get("security_vulnerabilities", {}).status == "passed" else "❌"
            
            report.append(f"| {component} | {unit_status} | {integration_status} | {performance_status} | {security_status} |")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by FraudGuard 360 Comprehensive Test Suite*")
        
        return "\n".join(report)
    
    def run_all_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run all or specified test suites"""
        logger.info("Starting comprehensive test suite...")
        self.start_time = time.time()
        
        # Check prerequisites
        prerequisites = self.check_prerequisites()
        missing_prereqs = [k for k, v in prerequisites.items() if not v]
        
        if missing_prereqs:
            logger.warning(f"Missing prerequisites: {missing_prereqs}")
            logger.warning("Some tests may fail or be skipped")
        
        # Setup test environment
        self.setup_test_environment()
        
        # Default to all test types if none specified
        if test_types is None:
            test_types = ['unit', 'integration', 'performance', 'security']
        
        # Run unit tests
        if 'unit' in test_types:
            logger.info("=== Running Unit Tests ===")
            for suite_name in ['unit_api_gateway', 'unit_ml_service', 'unit_frontend']:
                if suite_name in self.test_suites:
                    # Check prerequisites
                    suite = self.test_suites[suite_name]
                    prerequisite_failed = False
                    
                    for prereq in suite.prerequisites:
                        if self.test_suites[prereq].status == "failed":
                            suite.status = "skipped"
                            logger.info(f"⏭️ Skipping {suite.name} (prerequisite {prereq} failed)")
                            prerequisite_failed = True
                            break
                    
                    if not prerequisite_failed:
                        self.test_results[suite_name] = self.run_unit_tests(suite_name)
        
        # Run integration tests
        if 'integration' in test_types:
            logger.info("=== Running Integration Tests ===")
            self.test_results['integration_pipeline'] = self.run_integration_tests()
        
        # Run performance tests
        if 'performance' in test_types:
            logger.info("=== Running Performance Tests ===")
            self.test_results['performance_load'] = self.run_performance_tests()
        
        # Run security tests
        if 'security' in test_types:
            logger.info("=== Running Security Tests ===")
            self.test_results['security_vulnerabilities'] = self.run_security_tests()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report()
        
        # Save report
        report_file = f"test_reports/comprehensive_test_report_{self.test_session_id}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Comprehensive test report saved to: {report_file}")
        
        # Return summary
        passed_suites = sum(1 for suite in self.test_suites.values() if suite.status == "passed")
        total_suites = len([s for s in self.test_suites.values() if s.status != "not_run"])
        
        summary = {
            'session_id': self.test_session_id,
            'total_suites': total_suites,
            'passed_suites': passed_suites,
            'failed_suites': total_suites - passed_suites,
            'success_rate': (passed_suites / total_suites * 100) if total_suites > 0 else 0,
            'total_execution_time': self.end_time - self.start_time,
            'report_file': report_file,
            'prerequisites': prerequisites,
            'test_results': self.test_results
        }
        
        return summary

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FraudGuard 360 Comprehensive Test Runner")
    parser.add_argument(
        '--types', 
        nargs='*', 
        choices=['unit', 'integration', 'performance', 'security'],
        default=['unit', 'integration', 'performance', 'security'],
        help='Types of tests to run (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize and run test runner
    test_runner = FraudGuardTestRunner()
    
    try:
        results = test_runner.run_all_tests(args.types)
        
        # Print summary
        print("\n" + "="*80)
        print("FRAUDGUARD 360 - COMPREHENSIVE TEST RESULTS")
        print("="*80)
        print(f"Session ID: {results['session_id']}")
        print(f"Total Test Suites: {results['total_suites']}")
        print(f"Passed: {results['passed_suites']} ✅")
        print(f"Failed: {results['failed_suites']} ❌") 
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Total Execution Time: {results['total_execution_time']:.2f}s")
        print(f"Report: {results['report_file']}")
        print("="*80)
        
        # Exit with error code if any tests failed
        if results['failed_suites'] > 0:
            logger.error("Some tests failed!")
            sys.exit(1)
        else:
            logger.info("All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()