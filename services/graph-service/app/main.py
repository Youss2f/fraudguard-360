"""
FraudGuard 360° Graph Service
Neo4j abstraction layer for fraud detection graph queries

This service provides:
- High-performance graph queries for fraud detection
- Subgraph extraction for ML inference
- Network analysis and community detection
- Real-time graph updates from stream processing
- Caching layer for frequently accessed patterns
"""

import os
import asyncio
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, Driver
import redis.asyncio as redis
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
GRAPH_QUERIES = Counter(
    "graph_queries_total", "Total graph queries", ["query_type", "status"]
)
QUERY_LATENCY = Histogram(
    "graph_query_duration_seconds", "Graph query latency", ["query_type"]
)
SUBGRAPH_SIZE = Histogram(
    "subgraph_nodes_count",
    "Subgraph size distribution",
    buckets=[10, 50, 100, 500, 1000, 5000],
)

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fraudguard360")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes

# Global variables
neo4j_driver: Optional[Driver] = None
redis_client: Optional[redis.Redis] = None


# Pydantic models
class GraphNode(BaseModel):
    """Graph node representation"""

    id: str
    labels: List[str]
    properties: Dict[str, Any]


class GraphRelationship(BaseModel):
    """Graph relationship representation"""

    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]


class Subgraph(BaseModel):
    """Subgraph containing nodes and relationships"""

    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FraudPattern(BaseModel):
    """Detected fraud pattern"""

    pattern_id: str
    pattern_type: str
    confidence: float
    nodes_involved: List[str]
    description: str
    detected_at: datetime


class NetworkStats(BaseModel):
    """Network analysis statistics"""

    total_nodes: int
    total_relationships: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]
    average_degree: float
    clustering_coefficient: float


class CommunityDetectionResult(BaseModel):
    """Community detection result"""

    communities: List[List[str]]
    modularity: float
    algorithm: str
    parameters: Dict[str, Any]


async def get_redis_client():
    """Dependency to get Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(REDIS_URL)
    return redis_client


def get_neo4j_driver():
    """Dependency to get Neo4j driver"""
    global neo4j_driver
    if neo4j_driver is None:
        raise HTTPException(status_code=503, detail="Neo4j driver not initialized")
    return neo4j_driver


async def init_neo4j():
    """Initialize Neo4j connection"""
    global neo4j_driver
    try:
        logger.info("Connecting to Neo4j", uri=NEO4J_URI)
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

        # Test connection
        with neo4j_driver.session() as session:
            result = session.run("RETURN 'Connection successful' as message")
            message = result.single()["message"]
            logger.info("Neo4j connection established", message=message)

        # Create indexes for performance
        await create_indexes()

    except Exception as e:
        logger.error("Failed to connect to Neo4j", error=str(e))
        raise


async def create_indexes():
    """Create necessary indexes for performance"""
    global neo4j_driver
    indexes = [
        "CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.user_id)",
        "CREATE INDEX transaction_id_index IF NOT EXISTS FOR (t:Transaction) ON (t.transaction_id)",
        "CREATE INDEX timestamp_index IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
        "CREATE INDEX amount_index IF NOT EXISTS FOR (t:Transaction) ON (t.amount)",
        "CREATE INDEX fraud_score_index IF NOT EXISTS FOR (t:Transaction) ON (t.fraud_score)",
    ]

    with neo4j_driver.session() as session:
        for index_query in indexes:
            try:
                session.run(index_query)
                logger.info("Index created", query=index_query)
            except Exception as e:
                logger.warning("Index creation failed", query=index_query, error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FraudGuard 360° Graph Service")
    await init_neo4j()

    yield

    # Shutdown
    global neo4j_driver, redis_client
    if neo4j_driver:
        neo4j_driver.close()
    if redis_client:
        await redis_client.close()
    logger.info("Graph Service shutdown complete")


# FastAPI app initialization
app = FastAPI(
    title="FraudGuard 360° Graph Service",
    description="Neo4j abstraction layer for fraud detection graph queries",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def execute_query(
    query: str, parameters: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Execute Cypher query and return results"""
    driver = get_neo4j_driver()

    with driver.session() as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


async def get_cached_result(cache_key: str, redis_client: redis.Redis) -> Optional[Any]:
    """Get cached result from Redis"""
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            import json

            return json.loads(cached)  # Safe JSON deserialization
    except Exception as e:
        logger.warning("Cache retrieval failed", key=cache_key, error=str(e))
    return None


async def cache_result(
    cache_key: str, data: Any, redis_client: redis.Redis, ttl: int = CACHE_TTL
):
    """Cache result in Redis"""
    try:
        await redis_client.setex(cache_key, ttl, str(data))
    except Exception as e:
        logger.warning("Cache storage failed", key=cache_key, error=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            session.run("RETURN 1")
        return {"status": "healthy", "neo4j": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/subgraph/{user_id}", response_model=Subgraph)
async def get_user_subgraph(
    user_id: str,
    depth: int = Query(2, ge=1, le=4, description="Subgraph depth"),
    min_amount: float = Query(0, ge=0, description="Minimum transaction amount"),
    hours_back: int = Query(24, ge=1, le=168, description="Hours to look back"),
    redis_client: redis.Redis = Depends(get_redis_client),
):
    """
    Extract subgraph around a specific user for fraud analysis

    Returns nodes and relationships within specified depth from the user,
    filtered by transaction amount and time window.
    """
    start_time = time.time()
    cache_key = f"subgraph:{user_id}:{depth}:{min_amount}:{hours_back}"

    try:
        # Check cache first
        cached_result = await get_cached_result(cache_key, redis_client)
        if cached_result:
            logger.info("Subgraph cache hit", user_id=user_id)
            return Subgraph(**cached_result)

        logger.info(
            "Extracting user subgraph",
            user_id=user_id,
            depth=depth,
            min_amount=min_amount,
        )

        # Cypher query to extract subgraph
        query = """
        MATCH path = (u:User {user_id: $user_id})-[*1..$depth]-(connected)
        WHERE ALL(r IN relationships(path) WHERE 
            (r:TRANSACTS AND r.amount >= $min_amount AND 
             r.timestamp >= datetime() - duration({hours: $hours_back}))
            OR type(r) IN ['OWNS', 'LOCATED_AT', 'CONNECTED_TO']
        )
        WITH collect(DISTINCT u) + collect(DISTINCT connected) as nodes,
             collect(DISTINCT relationships(path)) as rels
        UNWIND nodes as n
        UNWIND rels as r_list
        UNWIND r_list as r
        RETURN 
            collect(DISTINCT {
                id: toString(id(n)),
                labels: labels(n),
                properties: properties(n)
            }) as nodes,
            collect(DISTINCT {
                id: toString(id(r)),
                type: type(r),
                start_node: toString(id(startNode(r))),
                end_node: toString(id(endNode(r))),
                properties: properties(r)
            }) as relationships
        """

        parameters = {
            "user_id": user_id,
            "depth": depth,
            "min_amount": min_amount,
            "hours_back": hours_back,
        }

        result = execute_query(query, parameters)

        if not result:
            raise HTTPException(
                status_code=404, detail="User not found or no connections"
            )

        subgraph_data = result[0]
        nodes = [GraphNode(**node) for node in subgraph_data.get("nodes", [])]
        relationships = [
            GraphRelationship(**rel) for rel in subgraph_data.get("relationships", [])
        ]

        subgraph = Subgraph(
            nodes=nodes,
            relationships=relationships,
            metadata={
                "user_id": user_id,
                "depth": depth,
                "extraction_time": datetime.now().isoformat(),
                "query_time_ms": (time.time() - start_time) * 1000,
            },
        )

        # Cache the result
        await cache_result(cache_key, subgraph.dict(), redis_client)

        # Record metrics
        GRAPH_QUERIES.labels(query_type="subgraph", status="success").inc()
        QUERY_LATENCY.labels(query_type="subgraph").observe(time.time() - start_time)
        SUBGRAPH_SIZE.observe(len(nodes))

        logger.info(
            "Subgraph extracted",
            user_id=user_id,
            nodes=len(nodes),
            relationships=len(relationships),
            query_time_ms=(time.time() - start_time) * 1000,
        )

        return subgraph

    except Exception as e:
        GRAPH_QUERIES.labels(query_type="subgraph", status="error").inc()
        logger.error("Subgraph extraction failed", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Subgraph extraction failed: {str(e)}"
        )


@app.get("/patterns/fraud", response_model=List[FraudPattern])
async def detect_fraud_patterns(
    confidence_threshold: float = Query(0.7, ge=0.0, le=1.0),
    hours_back: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=1000),
    redis_client: redis.Redis = Depends(get_redis_client),
):
    """
    Detect known fraud patterns in the graph

    Identifies suspicious patterns like:
    - Rapid transaction sequences
    - Circular money flows
    - Velocity anomalies
    - Geographically impossible transactions
    """
    start_time = time.time()
    cache_key = f"fraud_patterns:{confidence_threshold}:{hours_back}:{limit}"

    try:
        # Check cache
        cached_result = await get_cached_result(cache_key, redis_client)
        if cached_result:
            return [FraudPattern(**pattern) for pattern in cached_result]

        logger.info(
            "Detecting fraud patterns", confidence_threshold=confidence_threshold
        )

        # Query for rapid transaction sequences (velocity fraud)
        velocity_query = """
        MATCH (u:User)-[t:TRANSACTS]->(target)
        WHERE t.timestamp >= datetime() - duration({hours: $hours_back})
        WITH u, count(t) as tx_count, sum(t.amount) as total_amount,
             collect(t.timestamp) as timestamps
        WHERE tx_count > 10 AND total_amount > 10000
        WITH u, tx_count, total_amount, timestamps,
             reduce(intervals = [], i IN range(0, size(timestamps)-2) | 
                intervals + [duration.inSeconds(timestamps[i+1], timestamps[i]).seconds]) as intervals
        WHERE size(intervals) > 0 AND 
              reduce(avg = 0.0, interval IN intervals | avg + interval) / size(intervals) < 300
        RETURN 
            'velocity_fraud_' + u.user_id as pattern_id,
            'rapid_transactions' as pattern_type,
            0.85 as confidence,
            [u.user_id] as nodes_involved,
            'User executed ' + toString(tx_count) + ' transactions in rapid succession' as description,
            datetime() as detected_at
        LIMIT $limit
        """

        # Query for circular money flows
        circular_query = """
        MATCH path = (u1:User)-[:TRANSACTS*3..5]->(u1)
        WHERE ALL(r IN relationships(path) WHERE 
            r.timestamp >= datetime() - duration({hours: $hours_back}) AND
            r.amount > 1000)
        WITH path, 
             reduce(total = 0, r IN relationships(path) | total + r.amount) as total_flow,
             nodes(path) as involved_nodes
        WHERE total_flow > 5000
        RETURN 
            'circular_' + toString(id(involved_nodes[0])) as pattern_id,
            'circular_flow' as pattern_type,
            0.90 as confidence,
            [n.user_id | n IN involved_nodes WHERE n:User] as nodes_involved,
            'Circular money flow detected: $' + toString(total_flow) as description,
            datetime() as detected_at
        LIMIT $limit/2
        """

        parameters = {"hours_back": hours_back, "limit": limit}

        velocity_results = execute_query(velocity_query, parameters)
        circular_results = execute_query(circular_query, parameters)

        all_patterns = velocity_results + circular_results
        patterns = []

        for pattern_data in all_patterns:
            if pattern_data["confidence"] >= confidence_threshold:
                patterns.append(
                    FraudPattern(
                        pattern_id=pattern_data["pattern_id"],
                        pattern_type=pattern_data["pattern_type"],
                        confidence=pattern_data["confidence"],
                        nodes_involved=pattern_data["nodes_involved"],
                        description=pattern_data["description"],
                        detected_at=pattern_data["detected_at"],
                    )
                )

        # Cache results
        await cache_result(
            cache_key, [p.dict() for p in patterns], redis_client, ttl=60
        )

        # Record metrics
        GRAPH_QUERIES.labels(query_type="fraud_patterns", status="success").inc()
        QUERY_LATENCY.labels(query_type="fraud_patterns").observe(
            time.time() - start_time
        )

        logger.info(
            "Fraud patterns detected",
            count=len(patterns),
            query_time_ms=(time.time() - start_time) * 1000,
        )

        return patterns

    except Exception as e:
        GRAPH_QUERIES.labels(query_type="fraud_patterns", status="error").inc()
        logger.error("Fraud pattern detection failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Pattern detection failed: {str(e)}"
        )


@app.get("/network/stats", response_model=NetworkStats)
async def get_network_statistics(redis_client: redis.Redis = Depends(get_redis_client)):
    """Get overall network statistics"""
    start_time = time.time()
    cache_key = "network_stats"

    try:
        # Check cache
        cached_result = await get_cached_result(cache_key, redis_client)
        if cached_result:
            return NetworkStats(**cached_result)

        logger.info("Computing network statistics")

        stats_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]-()
        WITH labels(n) as node_labels, type(r) as rel_type
        RETURN 
            count(DISTINCT n) as total_nodes,
            count(DISTINCT r) / 2 as total_relationships,
            collect(DISTINCT node_labels[0]) as node_types,
            collect(DISTINCT rel_type) as relationship_types
        """

        result = execute_query(stats_query)[0]

        # Calculate additional statistics
        degree_query = """
        MATCH (n)-[r]-()
        WITH n, count(r) as degree
        RETURN avg(degree) as avg_degree
        """

        degree_result = execute_query(degree_query)
        avg_degree = degree_result[0]["avg_degree"] if degree_result else 0.0

        # Node type counts
        node_type_query = """
        MATCH (n)
        RETURN labels(n)[0] as node_type, count(n) as count
        """

        node_type_results = execute_query(node_type_query)
        node_type_counts = {r["node_type"]: r["count"] for r in node_type_results}

        # Relationship type counts
        rel_type_query = """
        MATCH ()-[r]-()
        RETURN type(r) as rel_type, count(r) / 2 as count
        """

        rel_type_results = execute_query(rel_type_query)
        rel_type_counts = {r["rel_type"]: r["count"] for r in rel_type_results}

        stats = NetworkStats(
            total_nodes=result["total_nodes"],
            total_relationships=result["total_relationships"],
            node_types=node_type_counts,
            relationship_types=rel_type_counts,
            average_degree=float(avg_degree),
            clustering_coefficient=0.0,  # Placeholder - complex calculation
        )

        # Cache for 10 minutes
        await cache_result(cache_key, stats.dict(), redis_client, ttl=600)

        GRAPH_QUERIES.labels(query_type="network_stats", status="success").inc()
        QUERY_LATENCY.labels(query_type="network_stats").observe(
            time.time() - start_time
        )

        logger.info("Network statistics computed", stats=stats.dict())

        return stats

    except Exception as e:
        GRAPH_QUERIES.labels(query_type="network_stats", status="error").inc()
        logger.error("Network statistics failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Statistics computation failed: {str(e)}"
        )


@app.post("/transactions")
async def create_transaction(
    transaction_data: Dict[str, Any],
    redis_client: redis.Redis = Depends(get_redis_client),
):
    """Create a new transaction in the graph"""
    try:
        logger.info(
            "Creating transaction",
            transaction_id=transaction_data.get("transaction_id"),
        )

        # Cypher query to create transaction and relationships
        query = """
        MERGE (u1:User {user_id: $source_user})
        MERGE (u2:User {user_id: $target_user})
        CREATE (t:Transaction {
            transaction_id: $transaction_id,
            amount: $amount,
            timestamp: datetime($timestamp),
            fraud_score: coalesce($fraud_score, 0.0),
            location: $location
        })
        CREATE (u1)-[:TRANSACTS]->(t)
        CREATE (t)-[:TARGETS]->(u2)
        RETURN t.transaction_id as created_id
        """

        result = execute_query(query, transaction_data)

        # Invalidate relevant caches
        user_cache_pattern = f"subgraph:{transaction_data.get('source_user')}:*"
        # Note: In production, implement proper cache invalidation

        logger.info("Transaction created", transaction_id=result[0]["created_id"])

        return {"status": "created", "transaction_id": result[0]["created_id"]}

    except Exception as e:
        logger.error("Transaction creation failed", error=str(e))
        raise HTTPException(
            status_code=500, detail=f"Transaction creation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
