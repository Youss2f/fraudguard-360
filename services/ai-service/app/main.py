"""
FraudGuard 360° AI Service
Real-time fraud detection using GraphSAGE neural networks

This service provides:
- Graph-based fraud scoring using GraphSAGE
- Real-time inference API
- Model versioning and A/B testing
- Performance monitoring and metrics
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
INFERENCE_REQUESTS = Counter('fraud_inference_requests_total', 'Total inference requests', ['model_version', 'status'])
INFERENCE_LATENCY = Histogram('fraud_inference_duration_seconds', 'Inference latency', ['model_version'])
FRAUD_SCORES = Histogram('fraud_scores_distribution', 'Distribution of fraud scores', buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Configuration
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.2.0")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/graphsage_fraud_detector.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model and Redis
model: Optional['GraphSAGEFraudDetector'] = None
redis_client: Optional[redis.Redis] = None


class GraphSAGEFraudDetector(torch.nn.Module):
    """
    GraphSAGE-based fraud detection model
    
    Architecture:
    - 2-layer GraphSAGE with 128 hidden dimensions
    - Dropout for regularization
    - Binary classification output
    - Supports heterogeneous node features
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 1, dropout: float = 0.2):
        super(GraphSAGEFraudDetector, self).__init__()
        
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        
        # Feature engineering layers
        self.user_embedding = torch.nn.Embedding(10000, 32)  # User embeddings
        self.location_embedding = torch.nn.Embedding(1000, 16)  # Location embeddings
        self.time_encoder = torch.nn.Linear(4, 16)  # Time features (hour, day, month, year)
        
    def forward(self, x, edge_index, user_ids=None, locations=None, time_features=None):
        """
        Forward pass through GraphSAGE layers
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges]
            user_ids: User ID mappings for embeddings
            locations: Location IDs for embeddings
            time_features: Temporal features
        """
        # Enhance node features with embeddings
        if user_ids is not None:
            user_emb = self.user_embedding(user_ids)
            x = torch.cat([x, user_emb], dim=-1)
            
        if locations is not None:
            loc_emb = self.location_embedding(locations)
            x = torch.cat([x, loc_emb], dim=-1)
            
        if time_features is not None:
            time_emb = self.time_encoder(time_features)
            x = torch.cat([x, time_emb], dim=-1)
        
        # GraphSAGE layers
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Classification layer
        out = self.classifier(x)
        return torch.sigmoid(out)


# Pydantic models for API
class NodeFeature(BaseModel):
    """Individual node in the transaction graph"""
    node_id: str
    node_type: str = Field(..., description="Type: user, transaction, account, device")
    features: List[float] = Field(..., description="Numerical feature vector")
    user_id: Optional[int] = None
    location_id: Optional[int] = None
    timestamp: Optional[str] = None


class GraphEdge(BaseModel):
    """Edge connecting two nodes in the graph"""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    edge_type: str = Field(..., description="Edge type: transacts, owns, located_at")
    weight: float = Field(default=1.0, description="Edge weight/strength")


class SubgraphData(BaseModel):
    """Subgraph data for fraud scoring"""
    nodes: List[NodeFeature] = Field(..., description="Graph nodes")
    edges: List[GraphEdge] = Field(..., description="Graph edges")
    target_nodes: List[str] = Field(..., description="Nodes to score for fraud")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class FraudScore(BaseModel):
    """Fraud score result"""
    node_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, CRITICAL")
    confidence: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: Dict[str, float]
    model_version: str


class BatchFraudResponse(BaseModel):
    """Batch fraud detection response"""
    scores: List[FraudScore]
    processing_time_ms: float
    model_version: str
    graph_stats: Dict[str, int]


async def get_redis_client():
    """Dependency to get Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(REDIS_URL)
    return redis_client


async def load_model():
    """Load the trained GraphSAGE model"""
    global model
    try:
        logger.info("Loading GraphSAGE model", model_path=MODEL_PATH, device=str(DEVICE))
        
        model = GraphSAGEFraudDetector()
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded successfully", 
                       model_version=checkpoint.get('version', 'unknown'),
                       accuracy=checkpoint.get('accuracy', 'unknown'))
        else:
            logger.warning("Model file not found, using randomly initialized weights", model_path=MODEL_PATH)
        
        model.to(DEVICE)
        model.eval()
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting FraudGuard 360° AI Service", version=MODEL_VERSION)
    await load_model()
    
    yield
    
    # Shutdown
    global redis_client
    if redis_client:
        await redis_client.close()
    logger.info("AI Service shutdown complete")


# FastAPI app initialization
app = FastAPI(
    title="FraudGuard 360° AI Service",
    description="Real-time fraud detection using GraphSAGE neural networks",
    version=MODEL_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_subgraph(subgraph: SubgraphData) -> Data:
    """Convert subgraph data to PyTorch Geometric format"""
    # Create node mapping
    node_to_idx = {node.node_id: idx for idx, node in enumerate(subgraph.nodes)}
    
    # Extract node features
    node_features = torch.tensor([node.features for node in subgraph.nodes], dtype=torch.float)
    
    # Convert edges to tensor format
    edge_list = []
    for edge in subgraph.edges:
        if edge.source in node_to_idx and edge.target in node_to_idx:
            edge_list.append([node_to_idx[edge.source], node_to_idx[edge.target]])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Extract additional features
    user_ids = torch.tensor([node.user_id or 0 for node in subgraph.nodes], dtype=torch.long)
    location_ids = torch.tensor([node.location_id or 0 for node in subgraph.nodes], dtype=torch.long)
    
    return Data(
        x=node_features,
        edge_index=edge_index,
        user_ids=user_ids,
        location_ids=location_ids,
        node_mapping=node_to_idx
    )


def calculate_risk_level(fraud_probability: float) -> str:
    """Calculate risk level based on fraud probability"""
    if fraud_probability >= 0.8:
        return "CRITICAL"
    elif fraud_probability >= 0.6:
        return "HIGH"
    elif fraud_probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "device": str(DEVICE),
        "model_loaded": model is not None
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/scores", response_model=BatchFraudResponse)
async def predict_fraud_scores(
    subgraph: SubgraphData,
    background_tasks: BackgroundTasks,
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """
    Predict fraud scores for nodes in a subgraph
    
    This endpoint accepts a subgraph of transactions and related entities,
    runs GraphSAGE inference, and returns fraud probabilities for specified target nodes.
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info("Processing fraud scoring request", 
                   num_nodes=len(subgraph.nodes), 
                   num_edges=len(subgraph.edges),
                   target_nodes=len(subgraph.target_nodes))
        
        # Preprocess the subgraph
        graph_data = preprocess_subgraph(subgraph)
        graph_data = graph_data.to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            fraud_probabilities = model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.user_ids,
                graph_data.location_ids
            ).cpu().numpy().flatten()
        
        # Generate results for target nodes
        scores = []
        for target_node_id in subgraph.target_nodes:
            if target_node_id in graph_data.node_mapping:
                idx = graph_data.node_mapping[target_node_id]
                prob = float(fraud_probabilities[idx])
                
                # Record fraud score distribution
                FRAUD_SCORES.observe(prob)
                
                scores.append(FraudScore(
                    node_id=target_node_id,
                    fraud_probability=prob,
                    risk_level=calculate_risk_level(prob),
                    confidence=min(abs(prob - 0.5) * 2, 1.0),  # Confidence based on distance from 0.5
                    contributing_factors={
                        "graph_connectivity": np.random.uniform(0.1, 0.9),  # Placeholder
                        "behavioral_anomaly": np.random.uniform(0.1, 0.9),  # Placeholder
                        "temporal_pattern": np.random.uniform(0.1, 0.9),   # Placeholder
                    },
                    model_version=MODEL_VERSION
                ))
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Record metrics
        INFERENCE_REQUESTS.labels(model_version=MODEL_VERSION, status="success").inc()
        INFERENCE_LATENCY.labels(model_version=MODEL_VERSION).observe(processing_time / 1000)
        
        # Cache results in Redis for performance
        background_tasks.add_task(cache_results, redis_client, scores)
        
        response = BatchFraudResponse(
            scores=scores,
            processing_time_ms=processing_time,
            model_version=MODEL_VERSION,
            graph_stats={
                "nodes": len(subgraph.nodes),
                "edges": len(subgraph.edges),
                "target_nodes": len(subgraph.target_nodes)
            }
        )
        
        logger.info("Fraud scoring completed", 
                   processing_time_ms=processing_time,
                   avg_fraud_score=np.mean([s.fraud_probability for s in scores]))
        
        return response
        
    except Exception as e:
        INFERENCE_REQUESTS.labels(model_version=MODEL_VERSION, status="error").inc()
        logger.error("Fraud scoring failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


async def cache_results(redis_client: redis.Redis, scores: List[FraudScore]):
    """Cache fraud scores in Redis for quick lookup"""
    try:
        for score in scores:
            key = f"fraud_score:{score.node_id}"
            value = {
                "probability": score.fraud_probability,
                "risk_level": score.risk_level,
                "timestamp": asyncio.get_event_loop().time()
            }
            await redis_client.setex(key, 300, str(value))  # Cache for 5 minutes
    except Exception as e:
        logger.warning("Failed to cache results", error=str(e))


@app.get("/scores/{node_id}")
async def get_cached_score(node_id: str, redis_client: redis.Redis = Depends(get_redis_client)):
    """Get cached fraud score for a node"""
    try:
        key = f"fraud_score:{node_id}"
        cached_result = await redis_client.get(key)
        
        if cached_result:
            return {"node_id": node_id, "cached_result": eval(cached_result)}
        else:
            raise HTTPException(status_code=404, detail="No cached score found")
            
    except Exception as e:
        logger.error("Failed to retrieve cached score", node_id=node_id, error=str(e))
        raise HTTPException(status_code=500, detail="Cache lookup failed")


@app.post("/model/reload")
async def reload_model():
    """Reload the ML model (for A/B testing or updates)"""
    try:
        await load_model()
        logger.info("Model reloaded successfully", version=MODEL_VERSION)
        return {"status": "success", "model_version": MODEL_VERSION}
    except Exception as e:
        logger.error("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)