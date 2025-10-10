"""
FraudGuard 360° - Core ML Service
Real-time fraud detection with GraphSAGE neural networks
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass
import redis
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransactionFeatures:
    """Transaction feature representation for ML models"""
    amount: float
    merchant_category: str
    transaction_time: datetime
    location_risk_score: float
    account_age_days: int
    avg_transaction_amount_30d: float
    transaction_count_24h: int
    velocity_score: float
    device_fingerprint: str
    ip_reputation_score: float
    merchant_risk_score: float
    time_since_last_transaction: int

@dataclass
class FraudPrediction:
    """Fraud prediction result"""
    transaction_id: str
    fraud_probability: float
    risk_level: str
    confidence_score: float
    explanation: Dict[str, float]
    model_version: str
    processing_time_ms: float

class GraphSAGEModel(nn.Module):
    """
    GraphSAGE neural network for fraud detection
    Learns representations of users and merchants in transaction networks
    """
    
    def __init__(self, num_features: int, hidden_dim: int = 256, num_layers: int = 3):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(num_features, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolutions with residual connections
        h = x
        for i, conv in enumerate(self.convs):
            h_new = F.relu(conv(h, edge_index))
            if i > 0 and h.size() == h_new.size():
                h = h + h_new  # Residual connection
            else:
                h = h_new
            h = F.dropout(h, training=self.training)
        
        # Global pooling for graph-level prediction
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = torch.mean(h, dim=0, keepdim=True)
        
        # Classification
        return self.classifier(h)

class RealTimeFraudDetector:
    """
    Real-time fraud detection engine
    Combines multiple ML models for robust fraud detection
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.graphsage_model = None
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.model_version = "1.0.0"
        
    async def initialize_models(self, training_data_path: str = None):
        """Initialize and train ML models"""
        logger.info("Initializing fraud detection models...")
        
        if training_data_path:
            await self._train_models(training_data_path)
        else:
            await self._load_pretrained_models()
        
        logger.info("Models initialized successfully")
    
    async def _train_models(self, data_path: str):
        """Train ML models on historical data"""
        # Load training data
        df = pd.read_csv(data_path)
        
        # Feature engineering
        X, y = self._engineer_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        logger.info("Training Isolation Forest...")
        self.isolation_forest.fit(X_train_scaled)
        
        logger.info("Training Random Forest...")
        self.random_forest.fit(X_train_scaled, y_train)
        
        # Initialize GraphSAGE
        self.graphsage_model = GraphSAGEModel(
            num_features=X_train.shape[1],
            hidden_dim=self.model_config.get('hidden_dim', 256)
        )
        
        # Save models
        await self._save_models()
        
        logger.info("Model training completed")
    
    async def _load_pretrained_models(self):
        """Load pre-trained models"""
        try:
            with open('models/isolation_forest.pkl', 'rb') as f:
                self.isolation_forest = pickle.load(f)
            
            with open('models/random_forest.pkl', 'rb') as f:
                self.random_forest = pickle.load(f)
            
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.graphsage_model = GraphSAGEModel(num_features=50)
            self.graphsage_model.load_state_dict(
                torch.load('models/graphsage_model.pth')
            )
            self.graphsage_model.eval()
            
        except FileNotFoundError:
            logger.warning("Pre-trained models not found, using default initialization")
            await self._initialize_default_models()
    
    async def _initialize_default_models(self):
        """Initialize models with default parameters"""
        # Create dummy data for initialization
        dummy_data = np.random.randn(1000, 50)
        dummy_labels = np.random.randint(0, 2, 1000)
        
        self.scaler.fit(dummy_data)
        self.isolation_forest.fit(dummy_data)
        self.random_forest.fit(dummy_data, dummy_labels)
        
        self.graphsage_model = GraphSAGEModel(num_features=50)
    
    def _engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Engineer features from raw transaction data"""
        features = []
        
        # Basic transaction features
        features.extend([
            df['amount'].fillna(0),
            df['merchant_category_encoded'].fillna(0),
            df['hour_of_day'].fillna(0),
            df['day_of_week'].fillna(0),
            df['is_weekend'].fillna(0),
        ])
        
        # Behavioral features
        features.extend([
            df['account_age_days'].fillna(0),
            df['avg_amount_30d'].fillna(0),
            df['transaction_count_24h'].fillna(0),
            df['velocity_score'].fillna(0),
        ])
        
        # Risk features
        features.extend([
            df['location_risk_score'].fillna(0),
            df['device_risk_score'].fillna(0),
            df['ip_reputation_score'].fillna(0),
            df['merchant_risk_score'].fillna(0),
        ])
        
        # Network features
        features.extend([
            df['centrality_score'].fillna(0),
            df['clustering_coefficient'].fillna(0),
            df['pagerank_score'].fillna(0),
        ])
        
        X = np.column_stack(features)
        y = df['is_fraud'].values if 'is_fraud' in df.columns else np.zeros(len(df))
        
        return X, y
    
    async def predict_fraud(self, transaction: TransactionFeatures) -> FraudPrediction:
        """
        Real-time fraud prediction for a single transaction
        """
        start_time = datetime.now()
        
        # Extract features
        features = await self._extract_features(transaction)
        features_scaled = self.scaler.transform([features])
        
        # Get predictions from multiple models
        isolation_score = self.isolation_forest.decision_function(features_scaled)[0]
        rf_proba = self.random_forest.predict_proba(features_scaled)[0][1]
        
        # Ensemble prediction
        fraud_probability = self._ensemble_prediction(isolation_score, rf_proba)
        
        # Determine risk level
        risk_level = self._get_risk_level(fraud_probability)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence(isolation_score, rf_proba)
        
        # Generate explanation
        explanation = await self._generate_explanation(features, transaction)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        prediction = FraudPrediction(
            transaction_id=transaction.device_fingerprint,  # Using as temp ID
            fraud_probability=fraud_probability,
            risk_level=risk_level,
            confidence_score=confidence_score,
            explanation=explanation,
            model_version=self.model_version,
            processing_time_ms=processing_time
        )
        
        # Cache prediction
        await self._cache_prediction(prediction)
        
        return prediction
    
    async def _extract_features(self, transaction: TransactionFeatures) -> List[float]:
        """Extract numerical features from transaction"""
        
        # Time-based features
        hour = transaction.transaction_time.hour
        day_of_week = transaction.transaction_time.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Amount features
        amount_log = np.log1p(transaction.amount)
        amount_normalized = min(transaction.amount / 10000, 1.0)
        
        # Behavioral features
        velocity_normalized = min(transaction.velocity_score / 100, 1.0)
        
        features = [
            amount_log,
            amount_normalized,
            hash(transaction.merchant_category) % 1000 / 1000,  # Category encoding
            hour / 24,
            day_of_week / 7,
            is_weekend,
            transaction.location_risk_score,
            transaction.account_age_days / 3650,  # Normalize to ~10 years
            transaction.avg_transaction_amount_30d / 1000,
            transaction.transaction_count_24h / 100,
            velocity_normalized,
            hash(transaction.device_fingerprint) % 1000 / 1000,
            transaction.ip_reputation_score,
            transaction.merchant_risk_score,
            transaction.time_since_last_transaction / 3600,  # Hours
        ]
        
        # Pad to expected feature count
        while len(features) < 50:
            features.append(0.0)
        
        return features[:50]  # Ensure exactly 50 features
    
    def _ensemble_prediction(self, isolation_score: float, rf_proba: float) -> float:
        """Combine predictions from multiple models"""
        # Convert isolation forest score to probability
        isolation_proba = 1 / (1 + np.exp(isolation_score))  # Sigmoid transform
        
        # Weighted ensemble
        weights = [0.3, 0.7]  # [isolation_forest, random_forest]
        ensemble_proba = (
            weights[0] * isolation_proba + 
            weights[1] * rf_proba
        )
        
        return min(max(ensemble_proba, 0.0), 1.0)
    
    def _get_risk_level(self, fraud_probability: float) -> str:
        """Determine risk level based on fraud probability"""
        if fraud_probability >= 0.8:
            return "CRITICAL"
        elif fraud_probability >= 0.6:
            return "HIGH"
        elif fraud_probability >= 0.4:
            return "MEDIUM"
        elif fraud_probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _calculate_confidence(self, isolation_score: float, rf_proba: float) -> float:
        """Calculate prediction confidence"""
        # Higher confidence when models agree
        isolation_proba = 1 / (1 + np.exp(isolation_score))
        agreement = 1 - abs(isolation_proba - rf_proba)
        
        # Base confidence from model certainty
        rf_certainty = max(rf_proba, 1 - rf_proba)
        iso_certainty = max(isolation_proba, 1 - isolation_proba)
        
        confidence = (agreement * 0.5 + (rf_certainty + iso_certainty) / 2 * 0.5)
        return min(max(confidence, 0.0), 1.0)
    
    async def _generate_explanation(self, features: List[float], transaction: TransactionFeatures) -> Dict[str, float]:
        """Generate explainable AI features for the prediction"""
        # Feature importance from random forest
        feature_names = [
            'amount_log', 'amount_normalized', 'merchant_category',
            'hour', 'day_of_week', 'is_weekend', 'location_risk',
            'account_age', 'avg_amount_30d', 'transaction_count_24h',
            'velocity_score', 'device_fingerprint', 'ip_reputation',
            'merchant_risk', 'time_since_last'
        ]
        
        try:
            importance = self.random_forest.feature_importances_[:len(feature_names)]
            feature_contribution = {
                name: float(features[i] * importance[i])
                for i, name in enumerate(feature_names)
            }
        except:
            # Fallback explanation
            feature_contribution = {
                'transaction_amount': transaction.amount / 10000,
                'location_risk': transaction.location_risk_score,
                'velocity_score': transaction.velocity_score / 100,
                'merchant_risk': transaction.merchant_risk_score,
                'ip_reputation': transaction.ip_reputation_score
            }
        
        return feature_contribution
    
    async def _cache_prediction(self, prediction: FraudPrediction):
        """Cache prediction result in Redis"""
        try:
            key = f"fraud_prediction:{prediction.transaction_id}"
            data = {
                'fraud_probability': prediction.fraud_probability,
                'risk_level': prediction.risk_level,
                'confidence_score': prediction.confidence_score,
                'timestamp': datetime.now().isoformat(),
                'model_version': prediction.model_version
            }
            
            await asyncio.to_thread(
                self.redis_client.setex,
                key, 3600, json.dumps(data)  # Cache for 1 hour
            )
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")
    
    async def _save_models(self):
        """Save trained models to disk"""
        import os
        os.makedirs('models', exist_ok=True)
        
        with open('models/isolation_forest.pkl', 'wb') as f:
            pickle.dump(self.isolation_forest, f)
        
        with open('models/random_forest.pkl', 'wb') as f:
            pickle.dump(self.random_forest, f)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        if self.graphsage_model:
            torch.save(self.graphsage_model.state_dict(), 'models/graphsage_model.pth')
    
    async def batch_predict(self, transactions: List[TransactionFeatures]) -> List[FraudPrediction]:
        """Batch prediction for multiple transactions"""
        predictions = []
        
        # Process in batches for efficiency
        batch_size = 100
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            batch_predictions = await asyncio.gather(*[
                self.predict_fraud(transaction) for transaction in batch
            ])
            predictions.extend(batch_predictions)
        
        return predictions

# Model configuration
DEFAULT_MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 3,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'early_stopping_patience': 10
}

async def main():
    """Initialize and test the fraud detection system"""
    detector = RealTimeFraudDetector(DEFAULT_MODEL_CONFIG)
    await detector.initialize_models()
    
    # Test transaction
    test_transaction = TransactionFeatures(
        amount=250.0,
        merchant_category="retail",
        transaction_time=datetime.now(),
        location_risk_score=0.3,
        account_age_days=730,
        avg_transaction_amount_30d=150.0,
        transaction_count_24h=3,
        velocity_score=25.0,
        device_fingerprint="device_12345",
        ip_reputation_score=0.8,
        merchant_risk_score=0.4,
        time_since_last_transaction=3600
    )
    
    prediction = await detector.predict_fraud(test_transaction)
    
    print(f"Fraud Probability: {prediction.fraud_probability:.3f}")
    print(f"Risk Level: {prediction.risk_level}")
    print(f"Confidence: {prediction.confidence_score:.3f}")
    print(f"Processing Time: {prediction.processing_time_ms:.1f}ms")
    print(f"Top Risk Factors: {sorted(prediction.explanation.items(), key=lambda x: x[1], reverse=True)[:3]}")

if __name__ == "__main__":
    asyncio.run(main())