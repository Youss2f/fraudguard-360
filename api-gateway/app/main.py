from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
import jwt
from .models.schemas import Graph, GNNResult, Node, Edge
from .services.kafka_producer import produce_to_kafka
from .auth import verify_token
from neo4j import GraphDatabase, Result
import logging
import json
import httpx

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

security = HTTPBearer()
neo4j_driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "password"))

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

def parse_neo4j_result(result: Result) -> Graph:
    nodes = []
    edges = []
    seen_nodes = set()
    for record in result:
        n = record['n']
        m = record['m']
        r = record['r']
        if n.id not in seen_nodes:
            nodes.append(Node(id=n['id'], label=n['label'], features=n['features'], riskScore=n['risk_score']))
            seen_nodes.add(n.id)
        if m.id not in seen_nodes:
            nodes.append(Node(id=m['id'], label=m['label'], features=m['features'], riskScore=m['risk_score']))
            seen_nodes.add(m.id)
        edges.append(Edge(source=r.start_node['id'], target=r.end_node['id'], weight=r['weight']))
    return Graph(nodes=nodes, edges=edges)

@app.get("/graph/{graph_id}", response_model=Graph, dependencies=[Depends(verify_token)])
async def get_graph(graph_id: str):
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n)-[r]->(m) WHERE n.graph_id = $id RETURN n, r, m",
                id=graph_id
            )
            return parse_neo4j_result(result)
    except Exception as e:
        logger.error(f"Neo4j query error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/graph/expand/{node_id}", response_model=Graph, dependencies=[Depends(verify_token)])
async def expand_graph(node_id: str):
    try:
        with neo4j_driver.session() as session:
            result = session.run(
                "MATCH (n)-[r*1..2]-(m) WHERE n.id = $id RETURN n, r, m",
                id=node_id
            )
            return parse_neo4j_result(result)
    except Exception as e:
        logger.error(f"Neo4j expand query error: {e}")
        raise HTTPException(status_code=500, detail="Database error")

@app.post("/graph/analyze", response_model=List[GNNResult])
async def analyze_graph(subscriber_ids: List[str], credentials: HTTPAuthorizationCredentials = Depends(security)):
    verify_token(credentials)
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post("http://ml-service:8001/score", json={"ids": subscriber_ids})
            res.raise_for_status()
            return res.json()
    except httpx.HTTPError as e:
        logger.error(f"AI service error: {e}")
        raise HTTPException(status_code=503, detail="AI service unavailable")

@app.websocket("/ws/graph/{graph_id}")
async def websocket_endpoint(websocket: WebSocket, graph_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received: {data}")
            await websocket.send_text(f"Alert for graph {graph_id}: {data}")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.post("/graph/upload")
async def upload_graph(data: dict):
    try:
        produce_to_kafka("cdr-topic", data)
        return {"status": "uploaded"}
    except Exception as e:
        logger.error(f"Kafka produce error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(503, "Unhealthy")

@app.get("/metrics")
async def metrics():
    # Implement Prometheus metrics (assume instrumentator setup)
    return "metrics_placeholder"  # In real, use prometheus_fastapi_instrumentator
