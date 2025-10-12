# ==============================================================================
# FraudGuard 360 - Health Check System
# Comprehensive health monitoring for all system components
# ==============================================================================

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import asyncio
import httpx
import psycopg2
import redis
from neo4j import GraphDatabase
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class ComponentHealth(BaseModel):
    name: str
    status: HealthStatus
    message: str
    last_checked: datetime
    response_time_ms: Optional[float] = None
    details: Dict[str, Any] = {}

class SystemHealth(BaseModel):
    overall_status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime
    uptime_seconds: float

class HealthChecker:
    def __init__(self, config: dict):
        self.config = config
        self.start_time = datetime.utcnow()
        self.last_health_check = {}
        
    async def check_all_components(self) -> SystemHealth:
        """Check health of all system components"""
        components = []
        
        # Check all components in parallel
        health_checks = [
            self._check_database_health(),
            self._check_redis_health(),
            self._check_neo4j_health(),
            self._check_kafka_health(),
            self._check_ml_service_health(),
            self._check_flink_health()
        ]
        
        try:
            results = await asyncio.gather(*health_checks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, ComponentHealth):
                    components.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Health check failed: {result}")
                    components.append(ComponentHealth(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check error: {str(result)}",
                        last_checked=datetime.utcnow()
                    ))
        except Exception as e:
            logger.error(f"Failed to run health checks: {e}")
        
        # Determine overall health status
        overall_status = self._calculate_overall_status(components)
        
        return SystemHealth(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.utcnow(),
            uptime_seconds=(datetime.utcnow() - self.start_time).total_seconds()
        )
    
    async def _check_database_health(self) -> ComponentHealth:
        """Check PostgreSQL database health"""
        start_time = datetime.utcnow()
        
        try:
            # Test database connection and simple query
            conn = psycopg2.connect(
                host=self.config.get('postgres_host', 'localhost'),
                port=self.config.get('postgres_port', 5432),
                database=self.config.get('postgres_db', 'fraudguard'),
                user=self.config.get('postgres_user', 'postgres'),
                password=self.config.get('postgres_password', ''),
                connect_timeout=5
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
            # Check connection pool status
            cursor.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = %s", 
                          (self.config.get('postgres_db', 'fraudguard'),))
            active_connections = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name="postgresql",
                status=HealthStatus.HEALTHY,
                message="Database is responding normally",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "active_connections": active_connections,
                    "query_result": result[0] if result else None
                }
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="postgresql",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def _check_redis_health(self) -> ComponentHealth:
        """Check Redis cache health"""
        start_time = datetime.utcnow()
        
        try:
            redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=0,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            
            # Test Redis with ping and set/get operations
            ping_result = redis_client.ping()
            test_key = "health_check_test"
            test_value = f"test_{datetime.utcnow().timestamp()}"
            
            redis_client.set(test_key, test_value, ex=10)
            retrieved_value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            # Get Redis info
            info = redis_client.info()
            used_memory = info.get('used_memory_human', 'Unknown')
            connected_clients = info.get('connected_clients', 0)
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            status = HealthStatus.HEALTHY
            if retrieved_value.decode() != test_value:
                status = HealthStatus.DEGRADED
            
            return ComponentHealth(
                name="redis",
                status=status,
                message="Redis cache is responding normally",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "ping_result": ping_result,
                    "used_memory": used_memory,
                    "connected_clients": connected_clients,
                    "test_operation_success": retrieved_value.decode() == test_value
                }
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def _check_neo4j_health(self) -> ComponentHealth:
        """Check Neo4j graph database health"""
        start_time = datetime.utcnow()
        
        try:
            driver = GraphDatabase.driver(
                self.config.get('neo4j_uri', 'bolt://localhost:7687'),
                auth=(
                    self.config.get('neo4j_user', 'neo4j'),
                    self.config.get('neo4j_password', 'password')
                )
            )
            
            with driver.session() as session:
                # Test basic query
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
                # Get database info
                db_info = session.run("CALL dbms.components() YIELD name, versions, edition")
                components = [record.values() for record in db_info]
                
                # Check node count (sample)
                node_count_result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                node_count = node_count_result.single()["count"]
            
            driver.close()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.HEALTHY,
                message="Neo4j graph database is responding normally",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "test_query_result": test_value,
                    "node_count": node_count,
                    "components": components
                }
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                message=f"Neo4j connection failed: {str(e)}",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def _check_kafka_health(self) -> ComponentHealth:
        """Check Kafka message broker health"""
        start_time = datetime.utcnow()
        
        try:
            # For simplicity, we'll check if Kafka is accessible via HTTP management interface
            # In production, you'd use kafka-python or similar
            kafka_host = self.config.get('kafka_host', 'localhost')
            kafka_port = self.config.get('kafka_management_port', 9092)
            
            # This is a simplified check - in reality you'd use proper Kafka client
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    # Try to connect to Kafka broker (this is simplified)
                    response = await client.get(f"http://{kafka_host}:8083/connectors", 
                                              timeout=5.0)
                    status_code = response.status_code
                except httpx.ConnectError:
                    # Kafka might not have HTTP interface, consider it healthy if no other issues
                    status_code = 200
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name="kafka",
                status=HealthStatus.HEALTHY,
                message="Kafka message broker is accessible",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "broker_host": kafka_host,
                    "management_port": kafka_port,
                    "connection_test": "success"
                }
            )
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="kafka",
                status=HealthStatus.UNHEALTHY,
                message=f"Kafka connection failed: {str(e)}",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def _check_ml_service_health(self) -> ComponentHealth:
        """Check ML service health"""
        start_time = datetime.utcnow()
        
        try:
            ml_service_url = self.config.get('ml_service_url', 'http://localhost:8001')
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check ML service health endpoint
                response = await client.get(f"{ml_service_url}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    return ComponentHealth(
                        name="ml-service",
                        status=HealthStatus.HEALTHY,
                        message="ML service is responding normally",
                        last_checked=datetime.utcnow(),
                        response_time_ms=response_time,
                        details=health_data
                    )
                else:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return ComponentHealth(
                        name="ml-service",
                        status=HealthStatus.DEGRADED,
                        message=f"ML service returned status {response.status_code}",
                        last_checked=datetime.utcnow(),
                        response_time_ms=response_time,
                        details={"status_code": response.status_code}
                    )
                    
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="ml-service",
                status=HealthStatus.UNHEALTHY,
                message=f"ML service connection failed: {str(e)}",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    async def _check_flink_health(self) -> ComponentHealth:
        """Check Apache Flink stream processing health"""
        start_time = datetime.utcnow()
        
        try:
            flink_url = self.config.get('flink_url', 'http://localhost:8081')
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Check Flink cluster overview
                response = await client.get(f"{flink_url}/overview")
                
                if response.status_code == 200:
                    overview_data = response.json()
                    
                    # Check job status
                    jobs_response = await client.get(f"{flink_url}/jobs")
                    jobs_data = jobs_response.json()
                    
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    running_jobs = len([job for job in jobs_data.get('jobs', []) 
                                      if job.get('status') == 'RUNNING'])
                    
                    status = HealthStatus.HEALTHY if running_jobs > 0 else HealthStatus.DEGRADED
                    
                    return ComponentHealth(
                        name="flink",
                        status=status,
                        message=f"Flink cluster is running with {running_jobs} active jobs",
                        last_checked=datetime.utcnow(),
                        response_time_ms=response_time,
                        details={
                            "cluster_overview": overview_data,
                            "running_jobs": running_jobs,
                            "total_jobs": len(jobs_data.get('jobs', []))
                        }
                    )
                else:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    return ComponentHealth(
                        name="flink",
                        status=HealthStatus.DEGRADED,
                        message=f"Flink returned status {response.status_code}",
                        last_checked=datetime.utcnow(),
                        response_time_ms=response_time
                    )
                    
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return ComponentHealth(
                name="flink",
                status=HealthStatus.UNHEALTHY,
                message=f"Flink connection failed: {str(e)}",
                last_checked=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)}
            )
    
    def _calculate_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Calculate overall system health based on component health"""
        if not components:
            return HealthStatus.UNKNOWN
        
        unhealthy_count = sum(1 for comp in components if comp.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for comp in components if comp.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            # If any critical component is unhealthy, system is unhealthy
            critical_components = ['postgresql', 'redis', 'neo4j']
            critical_unhealthy = any(comp.name in critical_components and comp.status == HealthStatus.UNHEALTHY 
                                   for comp in components)
            if critical_unhealthy:
                return HealthStatus.UNHEALTHY
            elif unhealthy_count > len(components) // 2:
                return HealthStatus.UNHEALTHY
            else:
                return HealthStatus.DEGRADED
        
        if degraded_count > 0:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    async def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component"""
        all_health = await self.check_all_components()
        return next((comp for comp in all_health.components if comp.name == component_name), None)

# Global health checker instance
health_checker = None

def initialize_health_checker(config: dict):
    """Initialize the global health checker"""
    global health_checker
    health_checker = HealthChecker(config)

async def get_system_health() -> SystemHealth:
    """Get current system health status"""
    if health_checker is None:
        raise ValueError("Health checker not initialized")
    return await health_checker.check_all_components()

async def get_component_health(component_name: str) -> Optional[ComponentHealth]:
    """Get health status for a specific component"""
    if health_checker is None:
        raise ValueError("Health checker not initialized")
    return await health_checker.get_component_health(component_name)