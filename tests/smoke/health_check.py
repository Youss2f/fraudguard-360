#!/usr/bin/env python3
"""
Smoke tests for FraudGuard 360° health checks across environments
"""

import asyncio
import argparse
import httpx
import sys
import os
from typing import Dict, List, Optional


class HealthChecker:
    def __init__(self, environment: str):
        self.environment = environment
        self.base_urls = self._get_base_urls()
        self.timeout = 30.0
        
    def _get_base_urls(self) -> Dict[str, str]:
        """Get base URLs for different environments"""
        if self.environment == "staging":
            return {
                "api_gateway": os.getenv("STAGING_API_GATEWAY_URL", "https://staging-api.fraudguard360.com"),
                "ai_service": os.getenv("STAGING_AI_SERVICE_URL", "https://staging-ai.fraudguard360.com"), 
                "graph_service": os.getenv("STAGING_GRAPH_SERVICE_URL", "https://staging-graph.fraudguard360.com"),
                "frontend": os.getenv("STAGING_FRONTEND_URL", "https://staging.fraudguard360.com")
            }
        elif self.environment == "production":
            return {
                "api_gateway": os.getenv("PRODUCTION_API_GATEWAY_URL", "https://api.fraudguard360.com"),
                "ai_service": os.getenv("PRODUCTION_AI_SERVICE_URL", "https://ai.fraudguard360.com"),
                "graph_service": os.getenv("PRODUCTION_GRAPH_SERVICE_URL", "https://graph.fraudguard360.com"), 
                "frontend": os.getenv("PRODUCTION_FRONTEND_URL", "https://fraudguard360.com")
            }
        else:
            raise ValueError(f"Unknown environment: {self.environment}")

    async def check_service_health(self, service_name: str, base_url: str) -> Dict:
        """Check health of a single service"""
        health_url = f"{base_url}/health"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(health_url)
                
                return {
                    "service": service_name,
                    "url": health_url,
                    "status_code": response.status_code,
                    "healthy": response.status_code == 200,
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:200]
                }
                
        except httpx.TimeoutException:
            return {
                "service": service_name,
                "url": health_url,
                "status_code": None,
                "healthy": False,
                "response_time": None,
                "error": "Timeout"
            }
        except Exception as e:
            return {
                "service": service_name,
                "url": health_url,
                "status_code": None,
                "healthy": False,
                "response_time": None,
                "error": str(e)
            }

    async def check_api_endpoints(self) -> List[Dict]:
        """Test critical API endpoints"""
        api_base = self.base_urls["api_gateway"]
        
        endpoints_to_test = [
            {"path": "/api/v1/fraud/analyze", "method": "POST", "requires_auth": True},
            {"path": "/api/v1/auth/health", "method": "GET", "requires_auth": False},
            {"path": "/api/v1/metrics", "method": "GET", "requires_auth": True}
        ]
        
        results = []
        
        for endpoint in endpoints_to_test:
            url = f"{api_base}{endpoint['path']}"
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if endpoint["method"] == "GET":
                        response = await client.get(url)
                    else:
                        # For POST endpoints, send minimal test data
                        test_data = {"test": True} if not endpoint["requires_auth"] else None
                        response = await client.post(url, json=test_data)
                    
                    # For auth-required endpoints, 401 is expected without token
                    expected_healthy = not endpoint["requires_auth"] or response.status_code in [401, 403]
                    
                    results.append({
                        "endpoint": endpoint["path"],
                        "method": endpoint["method"],
                        "status_code": response.status_code,
                        "healthy": expected_healthy,
                        "response_time": response.elapsed.total_seconds()
                    })
                    
            except Exception as e:
                results.append({
                    "endpoint": endpoint["path"],
                    "method": endpoint["method"],
                    "status_code": None,
                    "healthy": False,
                    "error": str(e)
                })
        
        return results

    async def run_all_checks(self) -> Dict:
        """Run all health checks"""
        print(f"🔍 Running smoke tests for {self.environment} environment...")
        
        # Check service health endpoints
        health_tasks = [
            self.check_service_health(service, url) 
            for service, url in self.base_urls.items()
        ]
        
        health_results = await asyncio.gather(*health_tasks)
        
        # Check API endpoints
        api_results = await self.check_api_endpoints()
        
        # Compile results
        results = {
            "environment": self.environment,
            "health_checks": health_results,
            "api_checks": api_results,
            "overall_healthy": all(check["healthy"] for check in health_results + api_results)
        }
        
        return results

    def print_results(self, results: Dict):
        """Print formatted results"""
        print(f"\n📊 Smoke Test Results for {results['environment'].upper()}")
        print("=" * 60)
        
        # Health check results
        print("\n🏥 Service Health Checks:")
        for check in results["health_checks"]:
            status = "✅" if check["healthy"] else "❌"
            service = check["service"].ljust(15)
            response_time = f"{check.get('response_time', 0):.2f}s" if check.get('response_time') else "N/A"
            
            print(f"  {status} {service} | {check.get('status_code', 'N/A')} | {response_time}")
            
            if not check["healthy"] and check.get("error"):
                print(f"      Error: {check['error']}")
        
        # API endpoint results  
        print("\n🔌 API Endpoint Checks:")
        for check in results["api_checks"]:
            status = "✅" if check["healthy"] else "❌"
            endpoint = f"{check['method']} {check['endpoint']}".ljust(30)
            response_time = f"{check.get('response_time', 0):.2f}s" if check.get('response_time') else "N/A"
            
            print(f"  {status} {endpoint} | {check.get('status_code', 'N/A')} | {response_time}")
            
            if not check["healthy"] and check.get("error"):
                print(f"      Error: {check['error']}")
        
        # Overall status
        overall_status = "✅ PASSED" if results["overall_healthy"] else "❌ FAILED"
        print(f"\n🎯 Overall Status: {overall_status}")
        
        if not results["overall_healthy"]:
            print("\n⚠️  Some checks failed. Please investigate before proceeding.")


async def main():
    parser = argparse.ArgumentParser(description="Run FraudGuard 360° smoke tests")
    parser.add_argument(
        "--environment", 
        choices=["staging", "production"],
        required=True,
        help="Environment to test"
    )
    parser.add_argument(
        "--output-format",
        choices=["console", "json"],
        default="console", 
        help="Output format"
    )
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.environment)
    results = await checker.run_all_checks()
    
    if args.output_format == "console":
        checker.print_results(results)
    else:
        import json
        print(json.dumps(results, indent=2))
    
    # Exit with error code if any checks failed
    if not results["overall_healthy"]:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())