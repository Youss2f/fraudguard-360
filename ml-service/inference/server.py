from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from models.graphsage import GraphSAGE, calculate_risk
from torch_geometric.data import Data
import torch
from neo4j import GraphDatabase
from typing import List
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ScoreRequest(BaseModel):
    ids: List[str]

class ScoreResponse(BaseModel):
    subscriber_id: str
    risk_score: float
    community_id: str
    explanation: str

app = FastAPI()
model = GraphSAGE(4, 16, 8)
model_loaded = False

try:
    model.load_state_dict(torch.load('models/graphsage.pt'))
    model.eval()
    model_loaded = True
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.warning("Model file not found - running in mock mode for development")
    model_loaded = False

driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "password"))

def fetch_subgraphs(session, ids: List[str]):
    # Similar to fetch_graph_data, but filtered for ids
    # Implement query: MATCH (n:Subscriber WHERE n.id IN $ids)-[r:CALL*1..3]-(m)
    # Convert to Data
    # Placeholder implementation (expand as needed)
    features = []  # Fetch features
    edges = []  # Fetch edges
    return Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(edges, dtype=torch.long).t())

@app.post("/score", response_model=List[ScoreResponse])
async def score(request: ScoreRequest):
    try:
        if not model_loaded:
            # Mock response for development
            return [
                ScoreResponse(
                    subscriber_id=id_, 
                    risk_score=0.5 + (hash(id_) % 50) / 100.0,  # Mock risk score between 0.5-1.0
                    community_id="mock_community", 
                    explanation="Mock response - model not loaded"
                ) for id_ in request.ids
            ]
        
        with torch.no_grad():
            with driver.session() as session:
                data = fetch_subgraphs(session, request.ids)
                embeddings = model(data.x, data.edge_index)
                risks = calculate_risk(embeddings)
                return [
                    ScoreResponse(
                        subscriber_id=id_, 
                        risk_score=float(risk.item()), 
                        community_id="default", 
                        explanation="Anomaly based on embedding distance"
                    ) for id_, risk in zip(request.ids, risks)
                ]
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(500, "Inference failed")

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model_loaded, "mode": "production" if model_loaded else "development"}
