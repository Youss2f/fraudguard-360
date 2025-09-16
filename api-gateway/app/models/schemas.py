from pydantic import BaseModel, Field
from typing import List

class Node(BaseModel):
    id: str
    label: str
    features: List[float]
    riskScore: float = Field(alias="risk_score")

class Edge(BaseModel):
    source: str
    target: str
    weight: float

class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

class GNNResult(BaseModel):
    subscriberId: str = Field(alias="subscriber_id")
    riskScore: float = Field(alias="risk_score")
    communityId: str = Field(alias="community_id")
    explanation: str = ""  # Default empty
