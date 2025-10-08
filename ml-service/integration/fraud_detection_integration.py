import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CDRRecord:
    """CDR record data structure"""
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
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class FraudPrediction:
    """Fraud prediction result"""
    user_id: str
    fraud_probability: float
    risk_score: float
    fraud_type: str
    confidence: float
    model_predictions: Dict[str, float]
    risk_factors: List[str]
    timestamp: str
    recommendation: str
    alert_level: str

class SimpleModelEnsemble:
    """Simplified model ensemble for fraud detection"""
    
    def __init__(self, models_path="models/"):
        self.models_path = models_path
        self.models = {}
        self.feature_stats = {}
        self.fraud_patterns = {}
        self.load_models()
        
    def load_models(self):
        """Load or initialize simple models"""
        logger.info("Initializing fraud detection models...")
        
        # Simple rule-based models
        self.models['rule_based'] = self.create_rule_based_model()
        self.models['statistical'] = self.create_statistical_model()
        self.models['behavioral'] = self.create_behavioral_model()
        
        # Load any existing trained models
        model_files = ['ensemble_stats.pkl', 'fraud_patterns.pkl']
        for filename in model_files:
            filepath = os.path.join(self.models_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                        if filename == 'ensemble_stats.pkl':
                            self.feature_stats = data
                        elif filename == 'fraud_patterns.pkl':
                            self.fraud_patterns = data
                    logger.info(f"Loaded {filename}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def create_rule_based_model(self):
        """Create rule-based fraud detection model"""
        rules = {
            'high_cost_international': {
                'condition': lambda r: r.international_calls > 0 and r.call_cost > 10,
                'weight': 0.8,
                'fraud_type': 'International Fraud'
            },
            'excessive_calls': {
                'condition': lambda r: r.calls_per_day > 100,
                'weight': 0.7,
                'fraud_type': 'Call Volume Fraud'
            },
            'long_duration_night': {
                'condition': lambda r: r.night_calls > 0 and r.call_duration > 300,
                'weight': 0.6,
                'fraud_type': 'After-hours Fraud'
            },
            'frequent_location_changes': {
                'condition': lambda r: r.location_changes > 10,
                'weight': 0.7,
                'fraud_type': 'Location-based Fraud'
            },
            'suspicious_patterns': {
                'condition': lambda r: r.suspicious_patterns > 0,
                'weight': 0.9,
                'fraud_type': 'Behavioral Anomaly'
            },
            'high_cost_per_minute': {
                'condition': lambda r: (r.call_cost / (r.call_duration + 1e-6)) > 2.0,
                'weight': 0.5,
                'fraud_type': 'Cost Anomaly'
            }
        }
        return rules
    
    def create_statistical_model(self):
        """Create statistical anomaly detection model"""
        # Simple statistical thresholds (would be learned from data)
        thresholds = {
            'call_duration': {'mean': 120, 'std': 80, 'max_z_score': 3},
            'call_cost': {'mean': 2.5, 'std': 2.0, 'max_z_score': 3},
            'calls_per_day': {'mean': 15, 'std': 12, 'max_z_score': 2.5},
            'location_changes': {'mean': 2, 'std': 1.5, 'max_z_score': 2},
            'network_connections': {'mean': 8, 'std': 5, 'max_z_score': 2}
        }
        return thresholds
    
    def create_behavioral_model(self):
        """Create behavioral pattern detection model"""
        patterns = {
            'call_frequency_spike': {
                'check': lambda r: r.call_frequency_variance > 5,
                'weight': 0.4
            },
            'network_expansion': {
                'check': lambda r: r.network_connections > r.unique_numbers_called * 2,
                'weight': 0.3
            },
            'cost_pattern_anomaly': {
                'check': lambda r: r.call_cost > r.calls_per_day * 0.5,
                'weight': 0.5
            }
        }
        return patterns
    
    def predict_rule_based(self, record: CDRRecord) -> Tuple[float, Dict]:
        """Rule-based prediction"""
        rules = self.models['rule_based']
        triggered_rules = []
        total_weight = 0
        
        for rule_name, rule in rules.items():
            if rule['condition'](record):
                triggered_rules.append({
                    'rule': rule_name,
                    'weight': rule['weight'],
                    'fraud_type': rule['fraud_type']
                })
                total_weight += rule['weight']
        
        # Calculate fraud probability based on triggered rules
        fraud_prob = min(1.0, total_weight / 2.0)  # Normalize
        
        return fraud_prob, {
            'triggered_rules': triggered_rules,
            'total_weight': total_weight
        }
    
    def predict_statistical(self, record: CDRRecord) -> Tuple[float, Dict]:
        """Statistical anomaly prediction"""
        thresholds = self.models['statistical']
        anomalies = []
        total_z_score = 0
        
        for feature, stats in thresholds.items():
            if hasattr(record, feature):
                value = getattr(record, feature)
                z_score = abs(value - stats['mean']) / (stats['std'] + 1e-6)
                
                if z_score > stats['max_z_score']:
                    anomalies.append({
                        'feature': feature,
                        'value': value,
                        'z_score': z_score,
                        'threshold': stats['max_z_score']
                    })
                
                total_z_score += min(z_score, 5)  # Cap at 5
        
        # Convert z-scores to fraud probability
        fraud_prob = min(1.0, total_z_score / 15.0)  # Normalize
        
        return fraud_prob, {
            'anomalies': anomalies,
            'total_z_score': total_z_score
        }
    
    def predict_behavioral(self, record: CDRRecord) -> Tuple[float, Dict]:
        """Behavioral pattern prediction"""
        patterns = self.models['behavioral']
        detected_patterns = []
        total_weight = 0
        
        for pattern_name, pattern in patterns.items():
            if pattern['check'](record):
                detected_patterns.append({
                    'pattern': pattern_name,
                    'weight': pattern['weight']
                })
                total_weight += pattern['weight']
        
        fraud_prob = min(1.0, total_weight)
        
        return fraud_prob, {
            'detected_patterns': detected_patterns,
            'total_weight': total_weight
        }
    
    def predict_ensemble(self, record: CDRRecord) -> FraudPrediction:
        """Generate ensemble prediction"""
        # Get predictions from all models
        rule_prob, rule_details = self.predict_rule_based(record)
        stat_prob, stat_details = self.predict_statistical(record)
        behav_prob, behav_details = self.predict_behavioral(record)
        
        # Weighted ensemble
        weights = {'rule_based': 0.4, 'statistical': 0.3, 'behavioral': 0.3}
        ensemble_score = (
            rule_prob * weights['rule_based'] +
            stat_prob * weights['statistical'] +
            behav_prob * weights['behavioral']
        )
        
        model_predictions = {
            'rule_based': rule_prob,
            'statistical': stat_prob,
            'behavioral': behav_prob,
            'ensemble': ensemble_score
        }
        
        # Calculate risk score (0-100)
        risk_score = min(100, ensemble_score * 100)
        
        # Determine fraud type and risk factors
        fraud_type, risk_factors = self.analyze_fraud_pattern(record, model_predictions, rule_details)
        
        # Calculate confidence
        predictions_array = np.array([rule_prob, stat_prob, behav_prob])
        confidence = max(0.1, 1.0 - np.std(predictions_array))
        
        # Generate recommendation and alert level
        recommendation, alert_level = self.generate_recommendation(ensemble_score, risk_factors)
        
        return FraudPrediction(
            user_id=record.user_id,
            fraud_probability=ensemble_score,
            risk_score=risk_score,
            fraud_type=fraud_type,
            confidence=confidence,
            model_predictions=model_predictions,
            risk_factors=risk_factors,
            timestamp=datetime.now().isoformat(),
            recommendation=recommendation,
            alert_level=alert_level
        )
    
    def analyze_fraud_pattern(self, record: CDRRecord, predictions: Dict, rule_details: Dict) -> Tuple[str, List[str]]:
        """Analyze fraud pattern and identify risk factors"""
        risk_factors = []
        fraud_type = "General Anomaly"
        
        # Check triggered rules for fraud type
        triggered_rules = rule_details.get('triggered_rules', [])
        if triggered_rules:
            # Use the highest weighted rule's fraud type
            highest_weight_rule = max(triggered_rules, key=lambda x: x['weight'])
            fraud_type = highest_weight_rule['fraud_type']
        
        # Identify risk factors
        if record.call_duration > 300:
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
        
        if record.call_cost / (record.call_duration + 1e-6) > 2.0:
            risk_factors.append("High cost per minute")
        
        if record.network_connections > record.unique_numbers_called * 2:
            risk_factors.append("Unusual network expansion")
        
        return fraud_type, risk_factors
    
    def generate_recommendation(self, fraud_probability: float, risk_factors: List[str]) -> Tuple[str, str]:
        """Generate recommendation and alert level"""
        if fraud_probability > 0.8:
            alert_level = "CRITICAL"
            recommendation = "IMMEDIATE ACTION: Block account and investigate immediately"
        elif fraud_probability > 0.6:
            alert_level = "HIGH"
            recommendation = "HIGH RISK: Flag for manual review and monitor closely"
        elif fraud_probability > 0.4:
            alert_level = "MEDIUM"
            recommendation = "MEDIUM RISK: Increase monitoring and set alerts"
        elif fraud_probability > 0.2:
            alert_level = "LOW"
            recommendation = "LOW RISK: Continue normal monitoring"
        else:
            alert_level = "NORMAL"
            recommendation = "NORMAL: No action required"
        
        return recommendation, alert_level

class FraudDetectionIntegration:
    """Integration service for fraud detection with Flink"""
    
    def __init__(self):
        self.ensemble = SimpleModelEnsemble()
        self.database_path = "fraud_detection.db"
        self.prediction_cache = deque(maxlen=10000)
        self.alert_cache = deque(maxlen=1000)
        self.running = True
        
        # Initialize database
        self.init_database()
        
        # Performance metrics
        self.metrics = {
            'total_predictions': 0,
            'high_risk_predictions': 0,
            'alerts_generated': 0,
            'average_processing_time': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info("Fraud detection integration service initialized")
    
    def init_database(self):
        """Initialize SQLite database for storing predictions and alerts"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    fraud_probability REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    fraud_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    model_predictions TEXT NOT NULL,
                    risk_factors TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    processed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    fraud_probability REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    fraud_type TEXT NOT NULL,
                    risk_factors TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    status TEXT DEFAULT 'OPEN',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def process_cdr_record(self, cdr_data: Dict) -> FraudPrediction:
        """Process a single CDR record"""
        start_time = time.time()
        
        try:
            # Convert to CDR record
            cdr_record = CDRRecord(**cdr_data)
            
            # Generate prediction
            prediction = self.ensemble.predict_ensemble(cdr_record)
            
            # Store prediction
            self.store_prediction(prediction)
            
            # Update cache
            self.prediction_cache.append(asdict(prediction))
            
            # Generate alert if necessary
            if prediction.alert_level in ['HIGH', 'CRITICAL']:
                self.generate_alert(prediction)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.update_metrics(processing_time, prediction)
            
            logger.info(f"Processed CDR for user {prediction.user_id} - Risk: {prediction.risk_score:.1f}%")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing CDR record: {e}")
            raise
    
    def store_prediction(self, prediction: FraudPrediction):
        """Store prediction in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions (
                    user_id, fraud_probability, risk_score, fraud_type,
                    confidence, model_predictions, risk_factors,
                    recommendation, alert_level, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.user_id,
                prediction.fraud_probability,
                prediction.risk_score,
                prediction.fraud_type,
                prediction.confidence,
                json.dumps(prediction.model_predictions),
                json.dumps(prediction.risk_factors),
                prediction.recommendation,
                prediction.alert_level,
                prediction.timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    def generate_alert(self, prediction: FraudPrediction):
        """Generate fraud alert"""
        try:
            alert = {
                'user_id': prediction.user_id,
                'alert_level': prediction.alert_level,
                'fraud_probability': prediction.fraud_probability,
                'risk_score': prediction.risk_score,
                'fraud_type': prediction.fraud_type,
                'risk_factors': prediction.risk_factors,
                'recommendation': prediction.recommendation,
                'timestamp': prediction.timestamp
            }
            
            # Store alert in database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (
                    user_id, alert_level, fraud_probability, risk_score,
                    fraud_type, risk_factors, recommendation
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.user_id,
                prediction.alert_level,
                prediction.fraud_probability,
                prediction.risk_score,
                prediction.fraud_type,
                json.dumps(prediction.risk_factors),
                prediction.recommendation
            ))
            
            conn.commit()
            conn.close()
            
            # Add to cache
            self.alert_cache.append(alert)
            
            # Log alert
            logger.warning(f"FRAUD ALERT - {prediction.alert_level}: User {prediction.user_id} - {prediction.fraud_type}")
            
        except Exception as e:
            logger.error(f"Error generating alert: {e}")
    
    def update_metrics(self, processing_time: float, prediction: FraudPrediction):
        """Update performance metrics"""
        self.metrics['total_predictions'] += 1
        
        if prediction.alert_level in ['HIGH', 'CRITICAL']:
            self.metrics['high_risk_predictions'] += 1
            
        if prediction.alert_level in ['HIGH', 'CRITICAL']:
            self.metrics['alerts_generated'] += 1
        
        # Update average processing time
        current_avg = self.metrics['average_processing_time']
        total_predictions = self.metrics['total_predictions']
        self.metrics['average_processing_time'] = (
            (current_avg * (total_predictions - 1) + processing_time) / total_predictions
        )
        
        self.metrics['last_updated'] = datetime.now().isoformat()
    
    def get_user_risk_profile(self, user_id: str) -> Optional[Dict]:
        """Get latest risk profile for user"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE user_id = ? 
                ORDER BY processed_at DESC 
                LIMIT 1
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving risk profile: {e}")
            return None
    
    def get_active_alerts(self, limit: int = 100) -> List[Dict]:
        """Get active fraud alerts"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE status = 'OPEN' 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            logger.error(f"Error retrieving alerts: {e}")
            return []
    
    def get_system_metrics(self) -> Dict:
        """Get system performance metrics"""
        try:
            # Add database metrics
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Count total records
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total_predictions_db = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE status = "OPEN"')
            active_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            self.metrics.update({
                'total_predictions_db': total_predictions_db,
                'active_alerts': active_alerts,
                'cache_size': len(self.prediction_cache),
                'alert_cache_size': len(self.alert_cache)
            })
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
        
        return self.metrics
    
    def start_batch_processing(self, input_file: str):
        """Process batch CDR data from file"""
        logger.info(f"Starting batch processing from {input_file}")
        
        try:
            # Read data (assuming CSV format)
            df = pd.read_csv(input_file)
            total_records = len(df)
            
            logger.info(f"Processing {total_records} records...")
            
            processed = 0
            alerts_generated = 0
            
            for _, row in df.iterrows():
                try:
                    cdr_data = row.to_dict()
                    prediction = self.process_cdr_record(cdr_data)
                    
                    processed += 1
                    if prediction.alert_level in ['HIGH', 'CRITICAL']:
                        alerts_generated += 1
                    
                    if processed % 1000 == 0:
                        logger.info(f"Processed {processed}/{total_records} records")
                        
                except Exception as e:
                    logger.error(f"Error processing record: {e}")
                    continue
            
            logger.info(f"Batch processing completed: {processed} records processed, {alerts_generated} alerts generated")
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
    
    def simulate_real_time_processing(self, duration_seconds: int = 60):
        """Simulate real-time processing for testing"""
        logger.info(f"Starting real-time simulation for {duration_seconds} seconds")
        
        import random
        
        start_time = time.time()
        processed = 0
        
        while time.time() - start_time < duration_seconds and self.running:
            try:
                # Generate synthetic CDR record
                cdr_data = {
                    'user_id': f'user_{random.randint(1000, 9999)}',
                    'call_duration': random.lognormal(4, 1),
                    'call_cost': random.gamma(2, 0.5),
                    'calls_per_day': random.poisson(15),
                    'unique_numbers_called': random.poisson(8),
                    'international_calls': random.choice([0, 1]),
                    'night_calls': random.choice([0, 1]),
                    'weekend_calls': random.choice([0, 1]),
                    'call_frequency_variance': random.exponential(2),
                    'location_changes': random.poisson(2),
                    'avg_call_gap': random.exponential(1),
                    'network_connections': random.poisson(10),
                    'suspicious_patterns': random.choice([0, 1]) if random.random() < 0.1 else 0
                }
                
                # Occasionally inject high-risk patterns
                if random.random() < 0.05:  # 5% fraud rate
                    cdr_data.update({
                        'call_cost': random.uniform(15, 50),
                        'calls_per_day': random.randint(80, 200),
                        'international_calls': 1,
                        'suspicious_patterns': 1
                    })
                
                prediction = self.process_cdr_record(cdr_data)
                processed += 1
                
                # Sleep to simulate real-time processing
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in simulation: {e}")
                continue
        
        logger.info(f"Simulation completed: {processed} records processed")

def main():
    """Main function for testing the integration service"""
    print("🚀 Starting FraudGuard 360 ML Integration Service")
    print("=" * 60)
    
    # Initialize integration service
    integration = FraudDetectionIntegration()
    
    # Test with sample data
    sample_cdr = {
        'user_id': 'test_user_001',
        'call_duration': 450,  # Long call
        'call_cost': 25.0,     # High cost
        'calls_per_day': 150,  # High frequency
        'unique_numbers_called': 20,
        'international_calls': 1,  # International
        'night_calls': 1,      # Night call
        'weekend_calls': 0,
        'call_frequency_variance': 8.5,
        'location_changes': 12,  # Frequent changes
        'avg_call_gap': 0.5,
        'network_connections': 35,
        'suspicious_patterns': 1  # Suspicious
    }
    
    print("\n🧪 Testing with high-risk sample:")
    prediction = integration.process_cdr_record(sample_cdr)
    
    print(f"User ID: {prediction.user_id}")
    print(f"Fraud Probability: {prediction.fraud_probability:.3f}")
    print(f"Risk Score: {prediction.risk_score:.1f}%")
    print(f"Fraud Type: {prediction.fraud_type}")
    print(f"Alert Level: {prediction.alert_level}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Risk Factors: {', '.join(prediction.risk_factors)}")
    print(f"Recommendation: {prediction.recommendation}")
    
    print("\n📊 System Metrics:")
    metrics = integration.get_system_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n🔄 Running real-time simulation (30 seconds)...")
    integration.simulate_real_time_processing(30)
    
    print("\n📊 Final Metrics:")
    final_metrics = integration.get_system_metrics()
    for key, value in final_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n🚨 Active Alerts:")
    alerts = integration.get_active_alerts(10)
    for i, alert in enumerate(alerts[:5], 1):
        print(f"  {i}. User {alert['user_id']} - {alert['alert_level']} - Risk: {alert['risk_score']:.1f}%")
    
    print("\n✅ Integration service test completed successfully!")

if __name__ == "__main__":
    main()