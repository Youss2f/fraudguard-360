"""
GraphSAGE model for fraud detection in telecom networks.
This model learns node representations based on network structure and node features.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for learning node embeddings in the fraud detection graph.
    
    The model uses two SAGEConv layers followed by a classifier head to predict
    fraud probability for each node (user).
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        # Classifier head for fraud detection
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, 1),
            torch.nn.Sigmoid()
        )
        
        # Anomaly detection components
        self.anomaly_threshold = 0.5
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GraphSAGE model.
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for each node (optional)
            
        Returns:
            embeddings: Node embeddings [num_nodes, out_channels]
            fraud_probs: Fraud probabilities [num_nodes, 1]
        """
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        embeddings = x
        
        # Calculate fraud probabilities
        fraud_probs = self.classifier(embeddings)
        
        return embeddings, fraud_probs
    
    def predict_fraud_probability(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict fraud probability for nodes.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            
        Returns:
            Fraud probabilities for each node
        """
        self.eval()
        with torch.no_grad():
            _, fraud_probs = self.forward(x, edge_index)
            return fraud_probs.squeeze()
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings without fraud prediction.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            
        Returns:
            Node embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings, _ = self.forward(x, edge_index)
            return embeddings


class FraudDetectionModel:
    """
    High-level interface for fraud detection using GraphSAGE.
    """
    
    def __init__(self, in_channels: int = 16, hidden_channels: int = 64, 
                 out_channels: int = 32, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = GraphSAGE(in_channels, hidden_channels, out_channels).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.BCELoss()
        
        # Feature statistics for normalization
        self.feature_stats = None
        
    def prepare_features(self, user_data: Dict[str, Any]) -> torch.Tensor:
        """
        Prepare node features from user data.
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Feature tensor for the user
        """
        features = []
        
        # Basic user features
        features.extend([
            user_data.get('call_count', 0) / 1000.0,  # Normalized call count
            user_data.get('sms_count', 0) / 1000.0,   # Normalized SMS count  
            user_data.get('data_sessions', 0) / 100.0, # Normalized data sessions
            user_data.get('total_duration', 0) / 10000.0, # Normalized duration
            user_data.get('total_cost', 0) / 1000.0,   # Normalized cost
            user_data.get('unique_callees', 0) / 100.0, # Normalized unique callees
            user_data.get('international_calls', 0) / 100.0, # Normalized international
            user_data.get('premium_rate_calls', 0) / 10.0,   # Normalized premium
            user_data.get('night_calls_ratio', 0),     # Night calls ratio (already 0-1)
            user_data.get('call_frequency', 0) / 50.0, # Normalized frequency
            user_data.get('avg_call_duration', 0) / 300.0, # Normalized avg duration
            user_data.get('location_diversity', 0) / 20.0,  # Normalized locations
            user_data.get('device_diversity', 0) / 5.0,     # Normalized devices
            user_data.get('account_age_days', 0) / 3650.0,  # Normalized account age
            user_data.get('plan_cost', 0) / 200.0,          # Normalized plan cost
            1.0  # Bias term
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def train_step(self, data: Data, labels: torch.Tensor) -> float:
        """
        Perform one training step.
        
        Args:
            data: Graph data
            labels: True fraud labels
            
        Returns:
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        _, fraud_probs = self.model(data.x, data.edge_index)
        loss = self.criterion(fraud_probs.squeeze(), labels.float())
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: Data, labels: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data: Graph data
            labels: True fraud labels
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        with torch.no_grad():
            _, fraud_probs = self.model(data.x, data.edge_index)
            predictions = (fraud_probs.squeeze() > 0.5).float()
            
            # Calculate metrics
            accuracy = (predictions == labels.float()).float().mean().item()
            
            # Precision, Recall, F1 for fraud class
            true_positives = ((predictions == 1) & (labels == 1)).sum().item()
            false_positives = ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives = ((predictions == 0) & (labels == 1)).sum().item()
            
            precision = true_positives / (true_positives + false_positives + 1e-7)
            recall = true_positives / (true_positives + false_negatives + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'fraud_predictions': predictions.sum().item()
            }
    
    def predict_user_risk(self, user_features: torch.Tensor, 
                         graph_data: Data) -> Dict[str, float]:
        """
        Predict fraud risk for a specific user.
        
        Args:
            user_features: User feature vector
            graph_data: Graph context data
            
        Returns:
            Risk assessment results
        """
        self.model.eval()
        with torch.no_grad():
            # Add user to graph (temporarily)
            extended_features = torch.cat([graph_data.x, user_features.unsqueeze(0)], dim=0)
            
            embeddings, fraud_probs = self.model(extended_features, graph_data.edge_index)
            
            user_embedding = embeddings[-1]  # Last node is our user
            user_fraud_prob = fraud_probs[-1].item()
            
            # Calculate anomaly score based on embedding distance
            if len(embeddings) > 1:
                other_embeddings = embeddings[:-1]
                distances = torch.norm(other_embeddings - user_embedding, dim=1)
                anomaly_score = torch.sigmoid(distances.mean()).item()
            else:
                anomaly_score = 0.5
            
            # Combine fraud probability and anomaly score
            final_risk_score = 0.7 * user_fraud_prob + 0.3 * anomaly_score
            
            return {
                'fraud_probability': user_fraud_prob,
                'anomaly_score': anomaly_score,
                'final_risk_score': final_risk_score,
                'risk_level': self._get_risk_level(final_risk_score)
            }
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to categorical risk level."""
        if score >= 0.9:
            return 'critical'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def save_model(self, path: str):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_stats': self.feature_stats
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.feature_stats = checkpoint.get('feature_stats')
        logger.info(f"Model loaded from {path}")


def calculate_risk(embeddings: torch.Tensor, method: str = 'distance') -> torch.Tensor:
    """
    Calculate risk scores from node embeddings using various methods.
    
    Args:
        embeddings: Node embeddings [num_nodes, embedding_dim]
        method: Risk calculation method ('distance', 'isolation', 'density')
        
    Returns:
        Risk scores for each node [num_nodes]
    """
    if method == 'distance':
        # Distance from mean embedding
        mean_emb = embeddings.mean(dim=0)
        distances = torch.norm(embeddings - mean_emb, dim=1)
        scores = torch.sigmoid(distances)
        
    elif method == 'isolation':
        # Isolation-based anomaly detection
        # Nodes with embeddings far from their k-nearest neighbors
        distances = torch.cdist(embeddings, embeddings)
        k = min(5, len(embeddings) - 1)
        knn_distances = torch.topk(distances, k + 1, largest=False)[0][:, 1:]  # Exclude self
        isolation_scores = knn_distances.mean(dim=1)
        scores = torch.sigmoid(isolation_scores)
        
    elif method == 'density':
        # Local density-based anomaly detection
        distances = torch.cdist(embeddings, embeddings)
        # Local density is inverse of average distance to k nearest neighbors
        k = min(10, len(embeddings) - 1)
        knn_distances = torch.topk(distances, k + 1, largest=False)[0][:, 1:]
        densities = 1.0 / (knn_distances.mean(dim=1) + 1e-7)
        # Lower density = higher anomaly score
        scores = torch.sigmoid(-densities + densities.mean())
        
    else:
        raise ValueError(f"Unknown risk calculation method: {method}")
    
    return scores
