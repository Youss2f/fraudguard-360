"""
Training script for FraudGuard 360 GraphSAGE model.
This script trains the fraud detection model using telecom network data.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.optim import Adam
from neo4j import GraphDatabase
from models.graphsage import FraudDetectionModel, GraphSAGE
import numpy as np
import logging
import argparse
import os
from datetime import datetime
from typing import Tuple, List, Dict, Any
import json
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FraudTrainingPipeline:
    """Complete training pipeline for fraud detection model."""
    
    def __init__(self, neo4j_uri: str = "bolt://neo4j:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def fetch_training_data(self) -> Tuple[Data, torch.Tensor]:
        """
        Fetch training data from Neo4j and prepare it for training.
        
        Returns:
            graph_data: PyTorch Geometric Data object
            labels: Fraud labels for training
        """
        logger.info("Fetching training data from Neo4j...")
        
        try:
            with self.neo4j_driver.session() as session:
                # Query to get all users with their features and relationships
                query = """
                MATCH (u:User)
                OPTIONAL MATCH (u)-[r:MADE_CALL|USES_DEVICE|LOCATED_AT]-(connected)
                WITH u, collect(DISTINCT connected) as connections
                OPTIONAL MATCH (u)-[:HAS_ALERT]->(alert:FraudAlert)
                WHERE alert.severity IN ['high', 'critical']
                RETURN 
                    u.id as user_id,
                    properties(u) as user_props,
                    size(connections) as connection_count,
                    count(alert) > 0 as is_fraud
                """
                
                result = session.run(query)
                records = list(result)
                
                if not records:
                    logger.warning("No training data found. Generating synthetic data...")
                    return self._generate_synthetic_data()
                
                # Prepare node features and labels
                user_ids = []
                features = []
                labels = []
                
                for record in records:
                    user_ids.append(record['user_id'])
                    
                    # Extract user features
                    props = record['user_props'] or {}
                    user_features = self._extract_user_features(props)
                    features.append(user_features)
                    
                    # Extract label (fraud or not)
                    labels.append(1.0 if record['is_fraud'] else 0.0)
                
                # Get relationships between users
                edge_query = """
                MATCH (u1:User)-[r:MADE_CALL]-(u2:User)
                WHERE u1.id IN $user_ids AND u2.id IN $user_ids
                RETURN u1.id as source, u2.id as target, count(r) as call_count
                """
                
                edge_result = session.run(edge_query, user_ids=user_ids)
                user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
                
                edges = []
                for edge_record in edge_result:
                    source_idx = user_to_idx.get(edge_record['source'])
                    target_idx = user_to_idx.get(edge_record['target'])
                    
                    if source_idx is not None and target_idx is not None:
                        # Add edge in both directions (undirected graph)
                        edges.append([source_idx, target_idx])
                        edges.append([target_idx, source_idx])
                
                # Convert to tensors
                x = torch.stack(features).float()
                y = torch.tensor(labels).float()
                
                if edges:
                    edge_index = torch.tensor(edges).t().contiguous()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                
                graph_data = Data(x=x, edge_index=edge_index)
                
                logger.info(f"Loaded {len(user_ids)} users, {len(edges)} edges, "
                           f"{sum(labels)} fraud cases ({sum(labels)/len(labels)*100:.1f}%)")
                
                return graph_data, y
                
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            logger.info("Falling back to synthetic data generation...")
            return self._generate_synthetic_data()
    
    def _extract_user_features(self, props: Dict[str, Any]) -> torch.Tensor:
        """Extract and normalize user features from properties."""
        features = [
            props.get('call_count', 0) / 1000.0,
            props.get('sms_count', 0) / 1000.0,
            props.get('data_sessions', 0) / 100.0,
            props.get('total_duration', 0) / 10000.0,
            props.get('total_cost', 0) / 1000.0,
            props.get('unique_callees', 0) / 100.0,
            props.get('international_calls', 0) / 100.0,
            props.get('premium_rate_calls', 0) / 10.0,
            props.get('night_calls_ratio', 0),
            props.get('call_frequency', 0) / 50.0,
            props.get('avg_call_duration', 0) / 300.0,
            props.get('location_diversity', 0) / 20.0,
            props.get('device_diversity', 0) / 5.0,
            props.get('account_age_days', 365) / 3650.0,
            props.get('plan_cost', 50) / 200.0,
            1.0  # Bias term
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _generate_synthetic_data(self, num_users: int = 1000) -> Tuple[Data, torch.Tensor]:
        """Generate synthetic training data for development."""
        logger.info(f"Generating synthetic training data for {num_users} users...")
        
        # Generate random user features
        features = []
        labels = []
        
        for i in range(num_users):
            # Generate features with some correlation to fraud
            is_fraud = np.random.random() < 0.1  # 10% fraud rate
            
            if is_fraud:
                # Fraudulent users have higher activity, more international calls, etc.
                call_count = np.random.gamma(5, 100)  # Higher call volume
                international_calls = np.random.gamma(3, 20)  # More international
                premium_rate_calls = np.random.poisson(8)  # More premium rate
                night_calls_ratio = np.random.beta(3, 2)  # More night activity
                location_diversity = np.random.gamma(2, 5)  # More locations
                device_diversity = np.random.poisson(3) + 1  # More devices
            else:
                # Normal users
                call_count = np.random.gamma(2, 50)
                international_calls = np.random.gamma(1, 5)
                premium_rate_calls = np.random.poisson(1)
                night_calls_ratio = np.random.beta(1, 4)
                location_diversity = np.random.gamma(1, 2)
                device_diversity = 1
            
            # Calculate derived features
            total_duration = call_count * np.random.gamma(2, 100)
            total_cost = call_count * np.random.gamma(2, 2)
            unique_callees = min(call_count * 0.8, np.random.gamma(3, 10))
            call_frequency = call_count / 24  # Calls per hour
            avg_call_duration = total_duration / max(call_count, 1)
            
            user_features = [
                call_count / 1000.0,
                np.random.poisson(50) / 1000.0,  # SMS count
                np.random.poisson(10) / 100.0,   # Data sessions
                total_duration / 10000.0,
                total_cost / 1000.0,
                unique_callees / 100.0,
                international_calls / 100.0,
                premium_rate_calls / 10.0,
                night_calls_ratio,
                call_frequency / 50.0,
                avg_call_duration / 300.0,
                location_diversity / 20.0,
                device_diversity / 5.0,
                np.random.uniform(30, 3650) / 3650.0,  # Account age
                np.random.uniform(20, 200) / 200.0,     # Plan cost
                1.0  # Bias
            ]
            
            features.append(torch.tensor(user_features, dtype=torch.float32))
            labels.append(1.0 if is_fraud else 0.0)
        
        # Generate random graph structure
        edges = []
        edge_prob = 0.01  # Probability of connection between any two users
        
        for i in range(num_users):
            for j in range(i + 1, num_users):
                if np.random.random() < edge_prob:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
        
        # Convert to tensors
        x = torch.stack(features)
        y = torch.tensor(labels)
        edge_index = torch.tensor(edges).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        
        graph_data = Data(x=x, edge_index=edge_index)
        
        fraud_rate = sum(labels) / len(labels) * 100
        logger.info(f"Generated synthetic data: {num_users} users, {len(edges)} edges, "
                   f"{sum(labels)} fraud cases ({fraud_rate:.1f}%)")
        
        return graph_data, y
    
    def train_model(self, graph_data: Data, labels: torch.Tensor, 
                   epochs: int = 200, lr: float = 0.01, 
                   train_split: float = 0.8) -> Dict[str, Any]:
        """
        Train the fraud detection model.
        
        Args:
            graph_data: Training graph data
            labels: Fraud labels
            epochs: Number of training epochs
            lr: Learning rate
            train_split: Fraction of data for training
            
        Returns:
            Training history and metrics
        """
        logger.info("Starting model training...")
        
        # Initialize model
        self.model = FraudDetectionModel(
            in_channels=graph_data.x.shape[1],
            hidden_channels=64,
            out_channels=32,
            device=str(self.device)
        )
        
        # Move data to device
        graph_data = graph_data.to(self.device)
        labels = labels.to(self.device)
        
        # Split data
        num_nodes = graph_data.x.shape[0]
        num_train = int(num_nodes * train_split)
        
        indices = torch.randperm(num_nodes)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_auc': []
        }
        
        best_val_auc = 0.0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.model.train()
            self.model.optimizer.zero_grad()
            
            _, fraud_probs = self.model.model(graph_data.x, graph_data.edge_index)
            train_loss = self.model.criterion(fraud_probs[train_mask].squeeze(), labels[train_mask])
            
            train_loss.backward()
            self.model.optimizer.step()
            
            # Validation
            self.model.model.eval()
            with torch.no_grad():
                _, val_fraud_probs = self.model.model(graph_data.x, graph_data.edge_index)
                val_loss = self.model.criterion(val_fraud_probs[val_mask].squeeze(), labels[val_mask])
                
                # Calculate metrics
                train_preds = (fraud_probs[train_mask].squeeze() > 0.5).float()
                val_preds = (val_fraud_probs[val_mask].squeeze() > 0.5).float()
                
                train_acc = (train_preds == labels[train_mask]).float().mean()
                val_acc = (val_preds == labels[val_mask]).float().mean()
                
                # Calculate AUC if we have both classes
                val_labels_np = labels[val_mask].cpu().numpy()
                val_probs_np = val_fraud_probs[val_mask].squeeze().cpu().numpy()
                
                if len(np.unique(val_labels_np)) > 1:
                    val_auc = roc_auc_score(val_labels_np, val_probs_np)
                else:
                    val_auc = 0.5
            
            # Record history
            history['train_loss'].append(train_loss.item())
            history['val_loss'].append(val_loss.item())
            history['train_acc'].append(train_acc.item())
            history['val_acc'].append(val_acc.item())
            history['val_auc'].append(val_auc)
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                torch.save(self.model.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        # Load best model
        self.model.model.load_state_dict(torch.load('best_model.pt'))
        
        logger.info(f"Training completed. Best validation AUC: {best_val_auc:.4f}")
        
        return history
    
    def evaluate_model(self, graph_data: Data, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating model...")
        
        metrics = self.model.evaluate(graph_data, labels)
        
        logger.info(f"Final Evaluation Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save_model(self, model_path: str = "models/fraud_model.pt"):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    
    def close(self):
        """Close Neo4j connection."""
        if self.neo4j_driver:
            self.neo4j_driver.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train FraudGuard 360 ML model")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--neo4j-uri", default="bolt://neo4j:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-password", default="password", help="Neo4j password")
    parser.add_argument("--model-path", default="models/fraud_model.pt", help="Model save path")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    
    args = parser.parse_args()
    
    # Initialize training pipeline
    pipeline = FraudTrainingPipeline(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )
    
    try:
        # Fetch training data
        if args.synthetic:
            graph_data, labels = pipeline._generate_synthetic_data()
        else:
            graph_data, labels = pipeline.fetch_training_data()
        
        # Train model
        history = pipeline.train_model(
            graph_data, labels, 
            epochs=args.epochs, 
            lr=args.lr
        )
        
        # Evaluate model
        metrics = pipeline.evaluate_model(graph_data, labels)
        
        # Save model
        pipeline.save_model(args.model_path)
        
        # Save training history
        with open("training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
            node_map[m['id']] = idx
            nodes.append(m['id'])
            features.append(m['features'])
            idx += 1
        edges.append([node_map[n['id']], node_map[m['id']]])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def reconstruction_loss(embeddings, edge_index):
    # Simple link prediction loss for unsupervised
    pos_loss = -torch.log(torch.sigmoid((embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=1))).mean()
    neg_edge_index = negative_sampling(edge_index, embeddings.size(0))  # Need to import from torch_geometric.utils
    neg_loss = -torch.log(1 - torch.sigmoid((embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1))).mean()
    return pos_loss + neg_loss

with driver.session() as session:
    data = fetch_graph_data(session)

model = GraphSAGE(in_channels=4, hidden_channels=16, out_channels=8)
optimizer = Adam(model.parameters(), lr=0.01)

loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=128)

for epoch in range(100):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        embeddings = model(batch.x, batch.edge_index)
        loss = reconstruction_loss(embeddings, batch.edge_index)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    logger.info(f"Epoch {epoch}: Loss {total_loss / len(loader)}")

torch.save(model.state_dict(), 'models/graphsage.pt')
