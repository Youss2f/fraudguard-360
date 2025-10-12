"""
ML Inference Server for FraudGuard 360
Provides fraud prediction using GraphSAGE model on telecom network data.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from models.graphsage import FraudDetectionModel, calculate_risk
from torch_geometric.data import Data
import torch
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging
import asyncio
import json
import os
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class UserFeatureRequest(BaseModel):
    user_id: str
    features: Dict[str, Any] = Field(..., description="User features dictionary")

class BatchScoreRequest(BaseModel):
    user_ids: List[str] = Field(..., description="List of user IDs to score")
    include_explanation: bool = Field(default=True, description="Include risk explanation")

class RiskScoreResponse(BaseModel):
    user_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    final_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class BatchScoreResponse(BaseModel):
    scores: List[RiskScoreResponse]
    model_version: str
    processing_time_ms: float

class NetworkAnalysisRequest(BaseModel):
    user_id: str
    depth: int = Field(default=2, ge=1, le=4, description="Network traversal depth")

class NetworkAnalysisResponse(BaseModel):
    user_id: str
    network_risk_score: float
    connected_users: List[Dict[str, Any]]
    risk_communities: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    neo4j_connected: bool
    uptime_seconds: float

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard 360 ML Service",
    description="AI-powered fraud detection using GraphSAGE on telecom network data",
    version="1.0.0"
)

# Global variables
fraud_model = None
neo4j_driver = None
model_loaded = False
model_version = "1.0.0"
server_start_time = datetime.now()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MODEL_PATH = os.getenv("MODEL_PATH", "models/fraud_model.pt")

def initialize_model():
    """Initialize the fraud detection model with enhanced error handling."""
    global fraud_model, model_loaded
    
    try:
        # Detect available device for better performance
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        fraud_model = FraudDetectionModel(
            in_channels=16,
            hidden_channels=64,
            out_channels=32,
            device=device
        )
        
        if os.path.exists(MODEL_PATH):
            fraud_model.load_model(MODEL_PATH)
            logger.info(f"Fraud detection model loaded from {MODEL_PATH}")
        else:
            # Initialize with pre-trained weights or random initialization
            logger.warning(f"Model file not found at {MODEL_PATH}, initializing with random weights")
            fraud_model.initialize_random_weights()
        
        # Validate model by running a test prediction
        test_features = torch.randn((1, 16))
        test_prediction = fraud_model.predict(test_features)
        logger.info(f"Model validation successful: test prediction = {test_prediction}")
        
        model_loaded = True
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        model_loaded = False

def initialize_neo4j():
    """Initialize Neo4j connection."""
    global neo4j_driver
    
    try:
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # Test connection
        with neo4j_driver.session() as session:
            session.run("RETURN 1")
        logger.info(f"Connected to Neo4j at {NEO4J_URI}")
        
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        neo4j_driver = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting FraudGuard 360 ML Service...")
    initialize_model()
    initialize_neo4j()
    logger.info("ML Service startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if neo4j_driver:
        neo4j_driver.close()
    logger.info("ML Service shutdown completed")

def fetch_user_graph_data(user_ids: List[str], depth: int = 2) -> Data:
    """
    Fetch graph data for specified users from Neo4j.
    
    Args:
        user_ids: List of user IDs
        depth: Graph traversal depth
        
    Returns:
        PyTorch Geometric Data object
    """
    if not neo4j_driver:
        # Return mock data for development
        num_nodes = len(user_ids)
        features = torch.randn(num_nodes, 16)  # Random features for development
        edge_index = torch.tensor([[0, 1], [1, 0]]).t().contiguous() if num_nodes > 1 else torch.empty((2, 0), dtype=torch.long)
        return Data(x=features, edge_index=edge_index)
    
    try:
        with neo4j_driver.session() as session:
            # Query to get user network with features
            query = """
            MATCH (u:User)
            WHERE u.id IN $user_ids
            OPTIONAL MATCH (u)-[r:MADE_CALL|USES_DEVICE|LOCATED_AT*1..%d]-(connected)
            WITH collect(DISTINCT u) + collect(DISTINCT connected) as nodes
            UNWIND nodes as node
            WITH DISTINCT node
            OPTIONAL MATCH (node)-[rel]-(other)
            WHERE other IN nodes
            RETURN 
                collect(DISTINCT {
                    id: node.id,
                    type: labels(node)[0],
                    properties: properties(node)
                }) as nodes,
                collect(DISTINCT {
                    source: startNode(rel).id,
                    target: endNode(rel).id,
                    type: type(rel),
                    properties: properties(rel)
                }) as edges
            """ % depth
            
            result = session.run(query, user_ids=user_ids)
            record = result.single()
            
            if not record:
                return Data(x=torch.randn(len(user_ids), 16), 
                          edge_index=torch.empty((2, 0), dtype=torch.long))
            
            nodes = record["nodes"]
            edges = record["edges"]
            
            # Create node index mapping
            node_to_idx = {node["id"]: idx for idx, node in enumerate(nodes)}
            
            # Prepare node features
            features = []
            for node in nodes:
                if node["type"] == "User":
                    # Extract user features
                    props = node["properties"]
                    user_features = fraud_model.prepare_features({
                        'call_count': props.get('call_count', 0),
                        'sms_count': props.get('sms_count', 0),
                        'total_duration': props.get('total_duration', 0),
                        'total_cost': props.get('total_cost', 0),
                        'international_calls': props.get('international_calls', 0),
                        'premium_rate_calls': props.get('premium_rate_calls', 0),
                        'unique_callees': props.get('unique_callees', 0),
                        'account_age_days': props.get('account_age_days', 365),
                        'plan_cost': props.get('plan_cost', 50),
                        'call_frequency': props.get('call_frequency', 0),
                        'avg_call_duration': props.get('avg_call_duration', 0),
                        'night_calls_ratio': props.get('night_calls_ratio', 0),
                        'location_diversity': props.get('location_diversity', 1),
                        'device_diversity': props.get('device_diversity', 1),
                        'data_sessions': props.get('data_sessions', 0)
                    })
                    features.append(user_features)
                else:
                    # For non-user nodes, use simple features
                    features.append(torch.zeros(16))
            
            # Prepare edge indices
            edge_indices = []
            for edge in edges:
                if edge["source"] in node_to_idx and edge["target"] in node_to_idx:
                    source_idx = node_to_idx[edge["source"]]
                    target_idx = node_to_idx[edge["target"]]
                    edge_indices.append([source_idx, target_idx])
                    edge_indices.append([target_idx, source_idx])  # Make undirected
            
            if features:
                x = torch.stack(features)
            else:
                x = torch.randn(len(user_ids), 16)
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            return Data(x=x, edge_index=edge_index)
            
    except Exception as e:
        logger.error(f"Error fetching graph data: {e}")
        # Return minimal graph data
        return Data(x=torch.randn(len(user_ids), 16), 
                   edge_index=torch.empty((2, 0), dtype=torch.long))

@app.post("/predict", response_model=RiskScoreResponse)
async def predict_user_risk(request: UserFeatureRequest):
    """
    Predict fraud risk for a single user.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Prepare user features
        user_features = fraud_model.prepare_features(request.features)
        
        # Get graph context (user's network)
        graph_data = fetch_user_graph_data([request.user_id], depth=2)
        
        # Predict risk
        risk_result = fraud_model.predict_user_risk(user_features, graph_data)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        explanation = None
        if request.include_explanation if hasattr(request, 'include_explanation') else True:
            explanation = f"Risk assessment based on network analysis. "
            explanation += f"Fraud probability: {risk_result['fraud_probability']:.3f}, "
            explanation += f"Anomaly score: {risk_result['anomaly_score']:.3f}. "
            explanation += f"Processing time: {processing_time:.1f}ms"
        
        return RiskScoreResponse(
            user_id=request.user_id,
            fraud_probability=risk_result['fraud_probability'],
            anomaly_score=risk_result['anomaly_score'],
            final_risk_score=risk_result['final_risk_score'],
            risk_level=risk_result['risk_level'],
            confidence=min(risk_result['fraud_probability'] + 0.1, 1.0),
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Prediction error for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchScoreResponse)
async def batch_predict_risk(request: BatchScoreRequest):
    """
    Predict fraud risk for multiple users in batch.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Get graph data for all users
        graph_data = fetch_user_graph_data(request.user_ids, depth=2)
        
        scores = []
        for i, user_id in enumerate(request.user_ids):
            try:
                # For batch processing, use simplified features
                # In production, these would come from a feature store
                mock_features = {
                    'call_count': np.random.randint(10, 1000),
                    'sms_count': np.random.randint(5, 500),
                    'total_duration': np.random.randint(100, 10000),
                    'international_calls': np.random.randint(0, 50),
                    'premium_rate_calls': np.random.randint(0, 10),
                    'unique_callees': np.random.randint(5, 100),
                    'account_age_days': np.random.randint(30, 3650),
                }
                
                user_features = fraud_model.prepare_features(mock_features)
                risk_result = fraud_model.predict_user_risk(user_features, graph_data)
                
                explanation = None
                if request.include_explanation:
                    explanation = f"Batch prediction - Risk level: {risk_result['risk_level']}"
                
                scores.append(RiskScoreResponse(
                    user_id=user_id,
                    fraud_probability=risk_result['fraud_probability'],
                    anomaly_score=risk_result['anomaly_score'],
                    final_risk_score=risk_result['final_risk_score'],
                    risk_level=risk_result['risk_level'],
                    confidence=min(risk_result['fraud_probability'] + 0.1, 1.0),
                    explanation=explanation
                ))
                
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                # Add default low-risk score for failed users
                scores.append(RiskScoreResponse(
                    user_id=user_id,
                    fraud_probability=0.1,
                    anomaly_score=0.1,
                    final_risk_score=0.1,
                    risk_level="low",
                    confidence=0.5,
                    explanation=f"Error in processing: {str(e)}"
                ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchScoreResponse(
            scores=scores,
            model_version=model_version,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - server_start_time).total_seconds()
    neo4j_connected = neo4j_driver is not None
    
    # Test Neo4j connection
    if neo4j_driver:
        try:
            with neo4j_driver.session() as session:
                session.run("RETURN 1")
        except:
            neo4j_connected = False
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version,
        neo4j_connected=neo4j_connected,
        uptime_seconds=uptime
    )

@app.get("/model/info")
async def model_info():
    """Get model information."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": model_version,
        "model_type": "GraphSAGE",
        "input_features": 16,
        "hidden_dimensions": 64,
        "output_dimensions": 32,
        "framework": "PyTorch",
        "description": "Graph neural network for fraud detection in telecom networks"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
