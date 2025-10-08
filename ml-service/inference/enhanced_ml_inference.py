import asyncio
import json
import logging
import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import aioredis
import aiokafka
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for API
class CDRRecord(BaseModel):
    user_id: str
    call_duration: float
    call_cost: float
    calls_per_day: int
    unique_numbers_called: int
    international_calls: int
    night_calls: int
    weekend_calls: int
    call_frequency_variance: float
    location_changes: int
    avg_call_gap: float
    network_connections: int
    suspicious_patterns: int
    timestamp: Optional[str] = None

class FraudPrediction(BaseModel):
    user_id: str
    fraud_probability: float
    risk_score: float
    fraud_type: str
    confidence: float
    model_predictions: Dict[str, float]
    risk_factors: List[str]
    timestamp: str
    recommendation: str

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    prediction_time_ms: float
    last_updated: datetime

class GraphSAGEModel(nn.Module):
    """Advanced GraphSAGE model for fraud detection"""
    
    def __init__(self, num_features, hidden_dim=128, num_classes=2, dropout=0.3):
        super(GraphSAGEModel, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Graph convolution layers (simplified for inference)
        self.conv1 = nn.Linear(num_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv3 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Simplified forward pass for inference without graph structure
        h1 = torch.relu(self.conv1(x))
        h1 = self.dropout(h1)
        
        h2 = torch.relu(self.conv2(h1))
        h2 = self.dropout(h2)
        
        h3 = torch.relu(self.conv3(h2))
        h3 = self.dropout(h3)
        
        out = self.classifier(h3)
        return torch.softmax(out, dim=1)

class ModelEnsemble:
    """Ensemble of fraud detection models"""
    
    def __init__(self, models_path="models/"):
        self.models_path = models_path
        self.models = {}
        self.scaler = None
        self.feature_columns = []
        self.ensemble_weights = {
            'isolation_forest': 0.15,
            'random_forest': 0.25,
            'gradient_boosting': 0.25,
            'graphsage': 0.20,
            'behavioral_profiler': 0.15
        }
        self.metrics = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading fraud detection models...")
        
        try:
            # Load preprocessing objects
            self.scaler = joblib.load(os.path.join(self.models_path, 'feature_scaler.pkl'))
            
            with open(os.path.join(self.models_path, 'feature_names.json'), 'r') as f:
                self.feature_columns = json.load(f)
            
            # Load traditional ML models
            model_files = {
                'isolation_forest': 'isolation_forest_model.pkl',
                'random_forest': 'random_forest_model.pkl',
                'gradient_boosting': 'gradient_boosting_model.pkl',
                'behavioral_profiler': 'behavioral_clustering_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_path, filename)
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            
            # Load GraphSAGE model
            graphsage_path = os.path.join(self.models_path, 'graphsage_model.pth')
            graphsage_info_path = os.path.join(self.models_path, 'graphsage_info.json')
            
            if os.path.exists(graphsage_path) and os.path.exists(graphsage_info_path):
                with open(graphsage_info_path, 'r') as f:
                    model_info = json.load(f)
                
                graphsage_model = GraphSAGEModel(
                    num_features=model_info['num_features'],
                    hidden_dim=model_info['hidden_dim'],
                    num_classes=model_info['num_classes']
                )
                
                graphsage_model.load_state_dict(torch.load(graphsage_path, map_location='cpu'))
                graphsage_model.eval()
                
                self.models['graphsage'] = graphsage_model
                logger.info("Loaded GraphSAGE model")
            else:
                logger.warning("GraphSAGE model files not found")
            
            # Load cluster fraud rates for behavioral profiler
            cluster_rates_path = os.path.join(self.models_path, 'cluster_fraud_rates.csv')
            if os.path.exists(cluster_rates_path):
                self.cluster_fraud_rates = pd.read_csv(cluster_rates_path, index_col=0)
            else:
                logger.warning("Cluster fraud rates not found")
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_features(self, record: CDRRecord) -> np.ndarray:
        """Preprocess CDR record for model inference"""
        # Convert to dictionary
        data = record.dict()
        
        # Create derived features
        data['cost_per_minute'] = data['call_cost'] / (data['call_duration'] + 1e-6)
        data['calls_per_unique_number'] = data['calls_per_day'] / (data['unique_numbers_called'] + 1)
        data['location_change_rate'] = data['location_changes'] / (data['calls_per_day'] + 1)
        data['network_density'] = data['network_connections'] / (data['unique_numbers_called'] + 1)
        
        # Extract features in correct order
        features = np.array([data[col] for col in self.feature_columns])
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        return features_scaled[0]
    
    def predict_single_model(self, model_name: str, features: np.ndarray) -> Tuple[float, Dict]:
        """Get prediction from a single model"""
        model = self.models.get(model_name)
        if model is None:
            return 0.0, {'error': f'Model {model_name} not available'}
        
        start_time = time.time()
        
        try:
            if model_name == 'isolation_forest':
                # Anomaly score (higher = more normal, lower = more anomalous)
                anomaly_score = model.decision_function(features.reshape(1, -1))[0]
                # Convert to fraud probability (0-1)
                fraud_prob = max(0, min(1, 0.5 - anomaly_score / 2))
                details = {'anomaly_score': anomaly_score}
                
            elif model_name in ['random_forest', 'gradient_boosting']:
                # Classification probability
                fraud_prob = model.predict_proba(features.reshape(1, -1))[0, 1]
                prediction = model.predict(features.reshape(1, -1))[0]
                details = {'prediction': int(prediction)}
                
            elif model_name == 'graphsage':
                # Neural network prediction
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output = model(features_tensor)
                    fraud_prob = output[0, 1].item()
                details = {'neural_output': output[0].tolist()}
                
            elif model_name == 'behavioral_profiler':
                # Clustering-based prediction
                cluster = model.predict(features.reshape(1, -1))[0]
                if hasattr(self, 'cluster_fraud_rates') and cluster in self.cluster_fraud_rates.index:
                    fraud_prob = self.cluster_fraud_rates.loc[cluster, 'fraud_rate']
                else:
                    fraud_prob = 0.1  # Default low probability for unknown clusters
                details = {'cluster': int(cluster)}
                
            else:
                fraud_prob = 0.0
                details = {'error': f'Unknown model: {model_name}'}
            
            prediction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            details['prediction_time_ms'] = prediction_time
            
            return fraud_prob, details
            
        except Exception as e:
            logger.error(f"Error in {model_name} prediction: {e}")
            return 0.0, {'error': str(e)}
    
    def predict_ensemble(self, record: CDRRecord) -> FraudPrediction:
        """Generate ensemble prediction for fraud detection"""
        # Preprocess features
        features = self.preprocess_features(record)
        
        # Get predictions from all models
        model_predictions = {}
        prediction_details = {}
        
        for model_name in self.models.keys():
            fraud_prob, details = self.predict_single_model(model_name, features)
            model_predictions[model_name] = fraud_prob
            prediction_details[model_name] = details
        
        # Calculate weighted ensemble score
        ensemble_score = 0.0
        total_weight = 0.0
        
        for model_name, fraud_prob in model_predictions.items():
            weight = self.ensemble_weights.get(model_name, 0.1)
            ensemble_score += fraud_prob * weight
            total_weight += weight
        
        if total_weight > 0:
            ensemble_score /= total_weight
        
        # Calculate risk score (0-100)
        risk_score = min(100, ensemble_score * 100)
        
        # Determine fraud type and risk factors
        fraud_type, risk_factors = self.analyze_fraud_pattern(record, model_predictions)
        
        # Calculate confidence based on model agreement
        predictions_array = np.array(list(model_predictions.values()))
        confidence = 1.0 - np.std(predictions_array) if len(predictions_array) > 1 else 0.8
        
        # Generate recommendation
        recommendation = self.generate_recommendation(ensemble_score, risk_factors)
        
        return FraudPrediction(
            user_id=record.user_id,
            fraud_probability=ensemble_score,
            risk_score=risk_score,
            fraud_type=fraud_type,
            confidence=confidence,
            model_predictions=model_predictions,
            risk_factors=risk_factors,
            timestamp=datetime.now().isoformat(),
            recommendation=recommendation
        )
    
    def analyze_fraud_pattern(self, record: CDRRecord, predictions: Dict[str, float]) -> Tuple[str, List[str]]:
        """Analyze fraud pattern and identify risk factors"""
        risk_factors = []
        
        # Analyze call patterns
        if record.call_duration > 300:  # 5 minutes
            risk_factors.append("Unusually long call duration")
        
        if record.calls_per_day > 50:
            risk_factors.append("High call frequency")
        
        if record.call_cost > 10:
            risk_factors.append("High call cost")
        
        if record.international_calls:
            risk_factors.append("International calls detected")
        
        if record.night_calls:
            risk_factors.append("Night-time calling pattern")
        
        if record.location_changes > 5:
            risk_factors.append("Frequent location changes")
        
        if record.suspicious_patterns:
            risk_factors.append("Suspicious behavioral patterns")
        
        # Determine fraud type based on patterns
        if record.international_calls and record.call_cost > 5:
            fraud_type = "International Fraud"
        elif record.calls_per_day > 100:
            fraud_type = "Call Volume Fraud"
        elif record.location_changes > 10:
            fraud_type = "Location-based Fraud"
        elif record.night_calls and record.call_duration > 200:
            fraud_type = "After-hours Fraud"
        elif predictions.get('behavioral_profiler', 0) > 0.7:
            fraud_type = "Behavioral Anomaly"
        else:
            fraud_type = "General Fraud Pattern"
        
        return fraud_type, risk_factors
    
    def generate_recommendation(self, fraud_probability: float, risk_factors: List[str]) -> str:
        """Generate recommendation based on fraud assessment"""
        if fraud_probability > 0.8:
            return "IMMEDIATE ACTION: Block account and investigate immediately"
        elif fraud_probability > 0.6:
            return "HIGH RISK: Flag for manual review and monitor closely"
        elif fraud_probability > 0.4:
            return "MEDIUM RISK: Increase monitoring and set alerts"
        elif fraud_probability > 0.2:
            return "LOW RISK: Continue normal monitoring"
        else:
            return "NORMAL: No action required"

class RealTimeFraudDetector:
    """Real-time fraud detection service"""
    
    def __init__(self):
        self.ensemble = ModelEnsemble()
        self.redis_client = None
        self.kafka_consumer = None
        self.kafka_producer = None
        self.prediction_cache = deque(maxlen=10000)
        self.metrics_cache = {}
        
        # Background processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        
    async def initialize(self):
        """Initialize async connections"""
        # Initialize Redis
        self.redis_client = aioredis.from_url("redis://localhost:6379")
        
        # Initialize Kafka
        self.kafka_consumer = aiokafka.AIOKafkaConsumer(
            'cdr-events',
            bootstrap_servers='localhost:9092',
            group_id='fraud-detection-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        self.kafka_producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        await self.kafka_consumer.start()
        await self.kafka_producer.start()
        
        logger.info("Real-time fraud detector initialized")
    
    async def process_cdr_stream(self):
        """Process CDR events from Kafka stream"""
        try:
            async for message in self.kafka_consumer:
                cdr_data = message.value
                
                # Convert to CDR record
                cdr_record = CDRRecord(**cdr_data)
                
                # Generate prediction
                prediction = self.ensemble.predict_ensemble(cdr_record)
                
                # Cache prediction
                self.prediction_cache.append(prediction.dict())
                
                # Store in Redis for real-time access
                await self.redis_client.setex(
                    f"fraud_prediction:{cdr_record.user_id}",
                    3600,  # 1 hour TTL
                    json.dumps(prediction.dict())
                )
                
                # Send alert if high risk
                if prediction.fraud_probability > 0.6:
                    await self.send_fraud_alert(prediction)
                
                # Send to downstream systems
                await self.kafka_producer.send(
                    'fraud-alerts',
                    prediction.dict()
                )
                
        except Exception as e:
            logger.error(f"Error processing CDR stream: {e}")
    
    async def send_fraud_alert(self, prediction: FraudPrediction):
        """Send fraud alert to monitoring systems"""
        alert = {
            'alert_type': 'fraud_detection',
            'severity': 'high' if prediction.fraud_probability > 0.8 else 'medium',
            'user_id': prediction.user_id,
            'fraud_probability': prediction.fraud_probability,
            'risk_score': prediction.risk_score,
            'fraud_type': prediction.fraud_type,
            'risk_factors': prediction.risk_factors,
            'recommendation': prediction.recommendation,
            'timestamp': prediction.timestamp
        }
        
        # Send to alerting system
        await self.kafka_producer.send('system-alerts', alert)
        
        logger.warning(f"FRAUD ALERT: User {prediction.user_id} - Risk: {prediction.risk_score:.1f}%")
    
    async def get_user_risk_profile(self, user_id: str) -> Optional[Dict]:
        """Get cached risk profile for user"""
        try:
            cached_prediction = await self.redis_client.get(f"fraud_prediction:{user_id}")
            if cached_prediction:
                return json.loads(cached_prediction)
            return None
        except Exception as e:
            logger.error(f"Error retrieving risk profile: {e}")
            return None
    
    def get_system_metrics(self) -> Dict:
        """Get system performance metrics"""
        return {
            'total_predictions': len(self.prediction_cache),
            'high_risk_users': len([p for p in self.prediction_cache if p['fraud_probability'] > 0.6]),
            'average_risk_score': np.mean([p['risk_score'] for p in self.prediction_cache]) if self.prediction_cache else 0,
            'model_performance': self.ensemble.metrics,
            'last_updated': datetime.now().isoformat()
        }

# FastAPI application
app = FastAPI(title="FraudGuard 360 ML Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = RealTimeFraudDetector()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await detector.initialize()
    
    # Start background stream processing
    asyncio.create_task(detector.process_cdr_stream())
    
    logger.info("FraudGuard 360 ML Service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    detector.running = False
    if detector.kafka_consumer:
        await detector.kafka_consumer.stop()
    if detector.kafka_producer:
        await detector.kafka_producer.stop()
    if detector.redis_client:
        await detector.redis_client.close()
    
    logger.info("FraudGuard 360 ML Service stopped")

@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(record: CDRRecord):
    """Generate fraud prediction for a single CDR record"""
    try:
        prediction = detector.ensemble.predict_ensemble(record)
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/risk-profile")
async def get_user_risk_profile(user_id: str):
    """Get risk profile for a specific user"""
    profile = await detector.get_user_risk_profile(user_id)
    if profile:
        return profile
    else:
        raise HTTPException(status_code=404, detail="Risk profile not found")

@app.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    return detector.get_system_metrics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(detector.ensemble.models),
        "version": "1.0.0"
    }

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    def retrain_models():
        # This would trigger the training pipeline
        logger.info("Model retraining triggered")
        # Implementation would call the training script
    
    background_tasks.add_task(retrain_models)
    return {"message": "Model retraining triggered"}

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_ml_inference:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
        reload=False
    )