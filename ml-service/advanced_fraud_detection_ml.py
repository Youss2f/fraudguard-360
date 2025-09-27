import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import DBSCAN
import joblib
import sqlite3
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FraudPrediction:
    """Fraud prediction result"""
    user_id: str
    prediction: str
    confidence: float
    risk_score: float
    fraud_type: str
    evidence: Dict[str, Any]
    timestamp: datetime

class CDRFeatures(BaseModel):
    """CDR features for ML model"""
    user_id: str
    call_duration: float
    call_cost: float
    call_type: str
    time_of_day: int
    day_of_week: int
    location: str
    destination_type: str
    bytes_transmitted: int
    is_roaming: bool
    previous_calls_count: int
    avg_call_duration: float
    total_cost_last_hour: float
    unique_destinations_last_hour: int
    location_changes_last_hour: int

class GraphSAGEModel(nn.Module):
    """GraphSAGE model for fraud detection using network analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2, num_layers: int = 2):
        super(GraphSAGEModel, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        x = global_mean_pool(x, batch)
        return torch.log_softmax(x, dim=1)

class AdvancedFraudDetectionML:
    """Advanced ML-powered fraud detection system"""
    
    def __init__(self, db_path: str = "fraud_detection.db"):
        self.db_path = db_path
        self.redis_client = None
        
        # ML Models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.graphsage_model = None
        
        # Feature preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Model performance tracking
        self.model_metrics = {}
        self.feature_importance = {}
        
        # Initialize database
        self._init_database()
        
        # Load pre-trained models if available
        self._load_models()
        
        logger.info("Advanced Fraud Detection ML system initialized")
    
    async def initialize_redis(self):
        """Initialize Redis connection for caching"""
        try:
            self.redis_client = await aioredis.create_redis_pool('redis://localhost:6379')
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
    
    def _init_database(self):
        """Initialize SQLite database for storing ML data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for ML training data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                features TEXT NOT NULL,
                label INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fraud_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_score REAL NOT NULL,
                fraud_type TEXT NOT NULL,
                evidence TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL NOT NULL,
                precision_score REAL NOT NULL,
                recall_score REAL NOT NULL,
                f1_score REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            self.isolation_forest = joblib.load('models/isolation_forest.joblib')
            self.random_forest = joblib.load('models/random_forest.joblib')
            self.scaler = joblib.load('models/scaler.joblib')
            self.label_encoders = joblib.load('models/label_encoders.joblib')
            logger.info("Pre-trained models loaded successfully")
        except FileNotFoundError:
            logger.info("No pre-trained models found, will train new models")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models"""
        import os
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.isolation_forest, 'models/isolation_forest.joblib')
        joblib.dump(self.random_forest, 'models/random_forest.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        joblib.dump(self.label_encoders, 'models/label_encoders.joblib')
        logger.info("Models saved successfully")
    
    def extract_features(self, cdr_data: Dict) -> CDRFeatures:
        """Extract features from CDR data for ML models"""
        
        # Basic features from CDR
        user_id = cdr_data.get('caller_id', '')
        call_duration = float(cdr_data.get('duration', 0))
        call_cost = float(cdr_data.get('cost', 0))
        call_type = cdr_data.get('call_type', 'LOCAL')
        location = cdr_data.get('location_caller', '')
        bytes_transmitted = int(cdr_data.get('bytes_transmitted', 0))
        
        # Time-based features
        timestamp = datetime.fromisoformat(cdr_data.get('start_time', datetime.now().isoformat()))
        time_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Destination analysis
        callee_id = cdr_data.get('callee_id', '')
        destination_type = self._classify_destination(callee_id)
        
        # Roaming detection
        is_roaming = call_type in ['ROAMING', 'INTERNATIONAL']
        
        # Historical features (simplified - in production, query actual history)
        historical_features = self._get_historical_features(user_id, timestamp)
        
        return CDRFeatures(
            user_id=user_id,
            call_duration=call_duration,
            call_cost=call_cost,
            call_type=call_type,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            location=location,
            destination_type=destination_type,
            bytes_transmitted=bytes_transmitted,
            is_roaming=is_roaming,
            **historical_features
        )
    
    def _classify_destination(self, callee_id: str) -> str:
        """Classify destination number type"""
        if not callee_id:
            return 'UNKNOWN'
        
        if callee_id.startswith('+1900') or callee_id.startswith('+1976'):
            return 'PREMIUM'
        elif callee_id.startswith('+1'):
            return 'NATIONAL'
        elif callee_id.startswith('+'):
            return 'INTERNATIONAL'
        else:
            return 'LOCAL'
    
    def _get_historical_features(self, user_id: str, current_time: datetime) -> Dict:
        """Get historical features for user (simplified implementation)"""
        # In production, this would query actual database
        # For now, return default values
        return {
            'previous_calls_count': 10,
            'avg_call_duration': 120.0,
            'total_cost_last_hour': 15.0,
            'unique_destinations_last_hour': 3,
            'location_changes_last_hour': 1
        }
    
    def preprocess_features(self, features: CDRFeatures) -> np.ndarray:
        """Preprocess features for ML models"""
        
        # Convert to dictionary for easier manipulation
        feature_dict = asdict(features)
        
        # Encode categorical variables
        categorical_features = ['call_type', 'location', 'destination_type']
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                # Fit with some default values
                self.label_encoders[feature].fit(['LOCAL', 'NATIONAL', 'INTERNATIONAL', 'ROAMING', 'PREMIUM', 'UNKNOWN'])
            
            try:
                feature_dict[feature] = self.label_encoders[feature].transform([feature_dict[feature]])[0]
            except ValueError:
                # Handle unseen categories
                feature_dict[feature] = 0
        
        # Convert boolean to int
        feature_dict['is_roaming'] = int(feature_dict['is_roaming'])
        
        # Remove non-numeric features
        numeric_features = [k for k, v in feature_dict.items() if k != 'user_id' and isinstance(v, (int, float))]
        feature_vector = np.array([feature_dict[f] for f in numeric_features]).reshape(1, -1)
        
        return feature_vector
    
    async def detect_fraud(self, cdr_data: Dict) -> FraudPrediction:
        """Main fraud detection method using multiple ML models"""
        
        # Extract features
        features = self.extract_features(cdr_data)
        feature_vector = self.preprocess_features(features)
        
        # Scale features
        if hasattr(self.scaler, 'n_features_in_'):
            feature_vector_scaled = self.scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector
        
        # Ensemble predictions
        predictions = {}
        confidences = {}
        
        # Isolation Forest (Anomaly Detection)
        try:
            anomaly_score = self.isolation_forest.decision_function(feature_vector_scaled)[0]
            is_anomaly = self.isolation_forest.predict(feature_vector_scaled)[0] == -1
            predictions['isolation_forest'] = 'FRAUD' if is_anomaly else 'NORMAL'
            confidences['isolation_forest'] = abs(anomaly_score)
        except Exception as e:
            logger.warning(f"Isolation Forest prediction failed: {e}")
            predictions['isolation_forest'] = 'NORMAL'
            confidences['isolation_forest'] = 0.5
        
        # Random Forest Classification
        try:
            if hasattr(self.random_forest, 'predict_proba'):
                rf_proba = self.random_forest.predict_proba(feature_vector_scaled)[0]
                rf_pred = self.random_forest.predict(feature_vector_scaled)[0]
                predictions['random_forest'] = 'FRAUD' if rf_pred == 1 else 'NORMAL'
                confidences['random_forest'] = max(rf_proba)
            else:
                predictions['random_forest'] = 'NORMAL'
                confidences['random_forest'] = 0.5
        except Exception as e:
            logger.warning(f"Random Forest prediction failed: {e}")
            predictions['random_forest'] = 'NORMAL'
            confidences['random_forest'] = 0.5
        
        # Rule-based detection
        rule_result = self._apply_business_rules(features)
        predictions['business_rules'] = rule_result['prediction']
        confidences['business_rules'] = rule_result['confidence']
        
        # Ensemble decision
        final_prediction, final_confidence, risk_score = self._ensemble_decision(predictions, confidences)
        
        # Determine fraud type
        fraud_type = self._determine_fraud_type(features, final_prediction)
        
        # Create evidence
        evidence = {
            'predictions': predictions,
            'confidences': confidences,
            'features': asdict(features),
            'anomaly_indicators': self._get_anomaly_indicators(features)
        }
        
        # Create final prediction
        prediction_result = FraudPrediction(
            user_id=features.user_id,
            prediction=final_prediction,
            confidence=final_confidence,
            risk_score=risk_score,
            fraud_type=fraud_type,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
        # Store prediction
        await self._store_prediction(prediction_result)
        
        return prediction_result
    
    def _apply_business_rules(self, features: CDRFeatures) -> Dict[str, Any]:
        """Apply business rules for fraud detection"""
        
        risk_factors = []
        risk_score = 0.0
        
        # High cost calls
        if features.call_cost > 100:
            risk_factors.append('high_cost_call')
            risk_score += 0.3
        
        # Premium rate numbers
        if features.destination_type == 'PREMIUM':
            risk_factors.append('premium_rate_destination')
            risk_score += 0.4
        
        # Unusual time patterns
        if features.time_of_day < 6 or features.time_of_day > 22:
            risk_factors.append('unusual_time')
            risk_score += 0.2
        
        # High volume patterns
        if features.previous_calls_count > 50:
            risk_factors.append('high_volume')
            risk_score += 0.3
        
        # Roaming anomalies
        if features.is_roaming and features.call_cost > 50:
            risk_factors.append('expensive_roaming')
            risk_score += 0.3
        
        # Location changes
        if features.location_changes_last_hour > 5:
            risk_factors.append('rapid_location_changes')
            risk_score += 0.4
        
        prediction = 'FRAUD' if risk_score > 0.5 else 'NORMAL'
        confidence = min(risk_score, 1.0)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'risk_factors': risk_factors,
            'risk_score': risk_score
        }
    
    def _ensemble_decision(self, predictions: Dict, confidences: Dict) -> Tuple[str, float, float]:
        """Make ensemble decision from multiple model predictions"""
        
        # Weighted voting
        weights = {
            'isolation_forest': 0.3,
            'random_forest': 0.4,
            'business_rules': 0.3
        }
        
        fraud_score = 0.0
        total_weight = 0.0
        
        for model, prediction in predictions.items():
            if model in weights:
                weight = weights[model]
                confidence = confidences.get(model, 0.5)
                
                if prediction == 'FRAUD':
                    fraud_score += weight * confidence
                
                total_weight += weight
        
        if total_weight > 0:
            fraud_score /= total_weight
        
        # Final decision
        final_prediction = 'FRAUD' if fraud_score > 0.5 else 'NORMAL'
        final_confidence = fraud_score if final_prediction == 'FRAUD' else (1.0 - fraud_score)
        risk_score = fraud_score
        
        return final_prediction, final_confidence, risk_score
    
    def _determine_fraud_type(self, features: CDRFeatures, prediction: str) -> str:
        """Determine specific type of fraud based on features"""
        
        if prediction != 'FRAUD':
            return 'NONE'
        
        # Premium rate fraud
        if features.destination_type == 'PREMIUM':
            return 'PREMIUM_RATE_FRAUD'
        
        # SIM box fraud indicators
        if (features.call_type == 'INTERNATIONAL' and 
            features.call_duration < 60 and 
            features.previous_calls_count > 20):
            return 'SIM_BOX_FRAUD'
        
        # Velocity fraud
        if features.previous_calls_count > 30:
            return 'VELOCITY_FRAUD'
        
        # Roaming fraud
        if features.is_roaming and features.call_cost > 50:
            return 'ROAMING_FRAUD'
        
        # Account takeover
        if features.location_changes_last_hour > 3:
            return 'ACCOUNT_TAKEOVER'
        
        return 'GENERAL_FRAUD'
    
    def _get_anomaly_indicators(self, features: CDRFeatures) -> List[str]:
        """Get list of anomaly indicators"""
        
        indicators = []
        
        if features.call_cost > 100:
            indicators.append('high_cost')
        
        if features.destination_type == 'PREMIUM':
            indicators.append('premium_destination')
        
        if features.time_of_day < 6 or features.time_of_day > 22:
            indicators.append('unusual_time')
        
        if features.call_duration < 30:
            indicators.append('short_duration')
        
        if features.is_roaming:
            indicators.append('roaming_call')
        
        if features.location_changes_last_hour > 2:
            indicators.append('location_anomaly')
        
        return indicators
    
    async def _store_prediction(self, prediction: FraudPrediction):
        """Store prediction in database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO fraud_predictions 
            (user_id, prediction, confidence, risk_score, fraud_type, evidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            prediction.user_id,
            prediction.prediction,
            prediction.confidence,
            prediction.risk_score,
            prediction.fraud_type,
            json.dumps(prediction.evidence)
        ))
        
        conn.commit()
        conn.close()
    
    async def train_models(self, training_data: List[Dict]):
        """Train ML models with new data"""
        
        if not training_data:
            logger.warning("No training data provided")
            return
        
        logger.info(f"Training models with {len(training_data)} samples")
        
        # Extract features and labels
        features_list = []
        labels = []
        
        for data in training_data:
            features = self.extract_features(data['cdr'])
            feature_vector = self.preprocess_features(features)
            features_list.append(feature_vector.flatten())
            labels.append(data['label'])  # 0: normal, 1: fraud
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_train_scaled)
        
        # Train Random Forest
        self.random_forest.fit(X_train_scaled, y_train)
        
        # Evaluate models
        await self._evaluate_models(X_test_scaled, y_test)
        
        # Save models
        self._save_models()
        
        logger.info("Model training completed")
    
    async def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate trained models"""
        
        # Random Forest evaluation
        rf_predictions = self.random_forest.predict(X_test)
        rf_report = classification_report(y_test, rf_predictions, output_dict=True)
        
        # Store performance metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO model_performance 
            (model_name, accuracy, precision_score, recall_score, f1_score)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'random_forest',
            rf_report['accuracy'],
            rf_report['1']['precision'],
            rf_report['1']['recall'],
            rf_report['1']['f1-score']
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Random Forest performance: Accuracy={rf_report['accuracy']:.3f}")
    
    async def get_user_risk_profile(self, user_id: str) -> Dict:
        """Get comprehensive risk profile for a user"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent predictions
        cursor.execute("""
            SELECT prediction, confidence, risk_score, fraud_type, timestamp
            FROM fraud_predictions
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, (user_id,))
        
        predictions = cursor.fetchall()
        conn.close()
        
        if not predictions:
            return {
                'user_id': user_id,
                'risk_level': 'LOW',
                'fraud_probability': 0.0,
                'recent_alerts': 0,
                'fraud_types': [],
                'last_assessment': None
            }
        
        # Calculate risk metrics
        fraud_count = sum(1 for p in predictions if p[0] == 'FRAUD')
        avg_risk_score = np.mean([p[2] for p in predictions])
        fraud_types = list(set([p[3] for p in predictions if p[0] == 'FRAUD']))
        
        # Determine risk level
        if avg_risk_score > 0.7:
            risk_level = 'HIGH'
        elif avg_risk_score > 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'user_id': user_id,
            'risk_level': risk_level,
            'fraud_probability': avg_risk_score,
            'recent_alerts': fraud_count,
            'fraud_types': fraud_types,
            'last_assessment': predictions[0][4],
            'prediction_history': [
                {
                    'prediction': p[0],
                    'confidence': p[1],
                    'risk_score': p[2],
                    'fraud_type': p[3],
                    'timestamp': p[4]
                } for p in predictions
            ]
        }

# FastAPI application
app = FastAPI(title="Advanced Fraud Detection ML Service", version="1.0.0")

# Global ML service instance
ml_service = AdvancedFraudDetectionML()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await ml_service.initialize_redis()

@app.post("/detect-fraud")
async def detect_fraud_endpoint(cdr_data: Dict):
    """Endpoint for fraud detection"""
    try:
        prediction = await ml_service.detect_fraud(cdr_data)
        return asdict(prediction)
    except Exception as e:
        logger.error(f"Fraud detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-models")
async def train_models_endpoint(training_data: List[Dict]):
    """Endpoint for training ML models"""
    try:
        await ml_service.train_models(training_data)
        return {"status": "success", "message": "Models trained successfully"}
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user-risk-profile/{user_id}")
async def get_user_risk_profile_endpoint(user_id: str):
    """Endpoint for getting user risk profile"""
    try:
        profile = await ml_service.get_user_risk_profile(user_id)
        return profile
    except Exception as e:
        logger.error(f"Risk profile error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Advanced Fraud Detection ML"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)