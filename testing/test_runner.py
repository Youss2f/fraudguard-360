# FraudGuard 360 - Test Execution Scripts
# Automated test execution and reporting

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "test-results"
        self.results_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_unit_tests(self):
        """Run all unit tests"""
        logger.info("🧪 Running unit tests...")
        
        results = {
            "api_gateway": self._run_api_gateway_tests(),
            "ml_service": self._run_ml_service_tests(),
            "frontend": self._run_frontend_tests(),
            "flink_jobs": self._run_flink_tests()
        }
        
        return results
    
    def _run_api_gateway_tests(self):
        """Run API Gateway unit tests"""
        logger.info("Testing API Gateway...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/", "-v", "--cov=app", 
                "--cov-report=xml", "--cov-report=html",
                "--junit-xml=test-results.xml"
            ], 
            cwd=self.project_root / "api-gateway",
            capture_output=True, text=True, timeout=300)
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Test execution timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _run_ml_service_tests(self):
        """Run ML Service unit tests"""
        logger.info("Testing ML Service...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/", "-v", "--cov=.", 
                "--cov-report=xml", "--cov-report=html",
                "--junit-xml=test-results.xml"
            ], 
            cwd=self.project_root / "ml-service",
            capture_output=True, text=True, timeout=300)
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Test execution timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _run_frontend_tests(self):
        """Run Frontend unit tests"""
        logger.info("Testing Frontend...")
        
        try:
            # First ensure dependencies are installed
            subprocess.run(["npm", "ci"], 
                         cwd=self.project_root / "frontend",
                         check=True, timeout=180)
            
            result = subprocess.run([
                "npm", "test", "--", "--coverage", "--watchAll=false"
            ], 
            cwd=self.project_root / "frontend",
            capture_output=True, text=True, timeout=300)
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Test execution timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _run_flink_tests(self):
        """Run Flink job tests"""
        logger.info("Testing Flink Jobs...")
        
        try:
            result = subprocess.run([
                "mvn", "test", "-B"
            ], 
            cwd=self.project_root / "flink-jobs",
            capture_output=True, text=True, timeout=600)
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Test execution timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_integration_tests(self):
        """Run integration tests"""
        logger.info("🔗 Running integration tests...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "test_fraud_detection_pipeline.py", "-v",
                "--junit-xml=integration-test-results.xml"
            ], 
            cwd=self.project_root / "tests" / "e2e",
            capture_output=True, text=True, timeout=900)
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Integration test timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_load_tests(self):
        """Run load tests"""
        logger.info("🚀 Running load tests...")
        
        try:
            result = subprocess.run([
                "node", "load-test-runner.js", "suite"
            ], 
            cwd=self.project_root / "testing" / "load-testing",
            capture_output=True, text=True, timeout=1800)
            
            return {
                "status": "success" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Load test timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_security_tests(self):
        """Run security tests"""
        logger.info("🔒 Running security tests...")
        
        try:
            # Run vulnerability scanner
            vuln_result = subprocess.run([
                sys.executable, "vulnerability_scanner.py",
                "--project-root", str(self.project_root),
                "--output", "vulnerability-results"
            ], 
            cwd=self.project_root / "testing" / "security-testing",
            capture_output=True, text=True, timeout=600)
            
            # Run security tester
            sec_result = subprocess.run([
                sys.executable, "security_tester.py",
                "--url", "http://localhost:8000",
                "--output", "security-results"
            ], 
            cwd=self.project_root / "testing" / "security-testing",
            capture_output=True, text=True, timeout=300)
            
            return {
                "vulnerability_scan": {
                    "status": "success" if vuln_result.returncode == 0 else "failed",
                    "return_code": vuln_result.returncode,
                    "stdout": vuln_result.stdout,
                    "stderr": vuln_result.stderr
                },
                "security_test": {
                    "status": "success" if sec_result.returncode == 0 else "failed",
                    "return_code": sec_result.returncode,
                    "stdout": sec_result.stdout,
                    "stderr": sec_result.stderr
                }
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Security test timed out"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        logger.info("🎯 Starting comprehensive test suite...")
        
        start_time = time.time()
        
        # Run all test categories
        results = {
            "timestamp": self.timestamp,
            "unit_tests": self.run_unit_tests(),
            "integration_tests": self.run_integration_tests(),
            "load_tests": self.run_load_tests(),
            "security_tests": self.run_security_tests()
        }
        
        end_time = time.time()
        results["execution_time"] = end_time - start_time
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        # Save results
        self._save_results(results)
        
        # Generate report
        self._generate_report(results)
        
        return results
    
    def _generate_summary(self, results):
        """Generate test summary"""
        summary = {
            "total_categories": 4,
            "passed_categories": 0,
            "failed_categories": 0,
            "overall_status": "unknown"
        }
        
        # Check unit tests
        unit_passed = all(
            test.get("status") == "success" 
            for test in results["unit_tests"].values()
        )
        if unit_passed:
            summary["passed_categories"] += 1
        else:
            summary["failed_categories"] += 1
        
        # Check integration tests
        if results["integration_tests"].get("status") == "success":
            summary["passed_categories"] += 1
        else:
            summary["failed_categories"] += 1
        
        # Check load tests
        if results["load_tests"].get("status") == "success":
            summary["passed_categories"] += 1
        else:
            summary["failed_categories"] += 1
        
        # Check security tests
        security_passed = (
            results["security_tests"].get("vulnerability_scan", {}).get("status") == "success" and
            results["security_tests"].get("security_test", {}).get("status") == "success"
        )
        if security_passed:
            summary["passed_categories"] += 1
        else:
            summary["failed_categories"] += 1
        
        # Overall status
        if summary["passed_categories"] == summary["total_categories"]:
            summary["overall_status"] = "success"
        elif summary["passed_categories"] > 0:
            summary["overall_status"] = "partial"
        else:
            summary["overall_status"] = "failed"
        
        return summary
    
    def _save_results(self, results):
        """Save test results to JSON file"""
        results_file = self.results_dir / f"test_results_{self.timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📁 Test results saved to {results_file}")
    
    def _generate_report(self, results):
        """Generate human-readable test report"""
        report_file = self.results_dir / f"test_report_{self.timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# FraudGuard 360 - Test Execution Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Execution Time:** {results['execution_time']:.2f} seconds\n")
            f.write(f"**Overall Status:** {results['summary']['overall_status'].upper()}\n\n")
            
            # Unit tests section
            f.write("## Unit Tests\n\n")
            for service, result in results["unit_tests"].items():
                status = "✅ PASSED" if result.get("status") == "success" else "❌ FAILED"
                f.write(f"- **{service.replace('_', ' ').title()}:** {status}\n")
            f.write("\n")
            
            # Integration tests section
            f.write("## Integration Tests\n\n")
            status = "✅ PASSED" if results["integration_tests"].get("status") == "success" else "❌ FAILED"
            f.write(f"- **End-to-End Pipeline:** {status}\n\n")
            
            # Load tests section
            f.write("## Load Tests\n\n")
            status = "✅ PASSED" if results["load_tests"].get("status") == "success" else "❌ FAILED"
            f.write(f"- **Performance Testing:** {status}\n\n")
            
            # Security tests section
            f.write("## Security Tests\n\n")
            vuln_status = "✅ PASSED" if results["security_tests"].get("vulnerability_scan", {}).get("status") == "success" else "❌ FAILED"
            sec_status = "✅ PASSED" if results["security_tests"].get("security_test", {}).get("status") == "success" else "❌ FAILED"
            f.write(f"- **Vulnerability Scan:** {vuln_status}\n")
            f.write(f"- **Security Testing:** {sec_status}\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- **Total Test Categories:** {results['summary']['total_categories']}\n")
            f.write(f"- **Passed Categories:** {results['summary']['passed_categories']}\n")
            f.write(f"- **Failed Categories:** {results['summary']['failed_categories']}\n")
            
            if results['summary']['overall_status'] == 'success':
                f.write("\n🎉 **All tests passed successfully!**\n")
            else:
                f.write("\n⚠️ **Some tests failed. Please review the detailed results.**\n")
        
        logger.info(f"📊 Test report generated: {report_file}")

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FraudGuard 360 Test Runner')
    parser.add_argument('--test-type', choices=['unit', 'integration', 'load', 'security', 'all'], 
                       default='all', help='Type of tests to run')
    parser.add_argument('--project-root', default='.', 
                       help='Project root directory')
    
    args = parser.parse_args()
    
    runner = TestRunner(project_root=args.project_root)
    
    try:
        if args.test_type == 'unit':
            results = runner.run_unit_tests()
        elif args.test_type == 'integration':
            results = runner.run_integration_tests()
        elif args.test_type == 'load':
            results = runner.run_load_tests()
        elif args.test_type == 'security':
            results = runner.run_security_tests()
        else:
            results = runner.run_all_tests()
        
        # Print summary
        if args.test_type == 'all':
            if results['summary']['overall_status'] == 'success':
                logger.info("🎉 All tests passed successfully!")
                sys.exit(0)
            else:
                logger.error(f"⚠️ Test failures detected: {results['summary']['failed_categories']}/{results['summary']['total_categories']} categories failed")
                sys.exit(1)
        else:
            # For individual test types, check the specific result
            success = False
            if args.test_type == 'unit':
                success = all(test.get("status") == "success" for test in results.values())
            else:
                success = results.get("status") == "success"
            
            if success:
                logger.info(f"✅ {args.test_type.title()} tests passed!")
                sys.exit(0)
            else:
                logger.error(f"❌ {args.test_type.title()} tests failed!")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()