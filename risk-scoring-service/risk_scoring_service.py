"""
FraudGuard 360° - Risk Scoring Service
Advanced risk assessment algorithms for real-time fraud prevention
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import logging
import json
import redis
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import httpx
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskFactors:
    """Comprehensive risk factors for scoring"""
    
    # Transaction Risk Factors
    amount_risk: float = 0.0
    velocity_risk: float = 0.0
    frequency_risk: float = 0.0
    pattern_risk: float = 0.0
    
    # Behavioral Risk Factors
    device_risk: float = 0.0
    location_risk: float = 0.0
    time_risk: float = 0.0
    merchant_risk: float = 0.0
    
    # Network Risk Factors
    ip_reputation_risk: float = 0.0
    social_network_risk: float = 0.0
    fraud_network_risk: float = 0.0
    
    # Account Risk Factors
    account_age_risk: float = 0.0
    credit_risk: float = 0.0
    identity_risk: float = 0.0
    kyc_risk: float = 0.0

@dataclass
class RiskScore:
    """Final risk assessment result"""
    user_id: str
    transaction_id: str
    overall_risk_score: float  # 0-100
    risk_category: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_factors: RiskFactors
    confidence_level: float
    recommendation: str
    explanation: str
    timestamp: datetime
    model_version: str

class BehavioralRiskAnalyzer:
    """
    Analyzes user behavioral patterns to detect anomalies
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        
    async def analyze_transaction_behavior(self, user_id: str, transaction_data: Dict) -> RiskFactors:
        """Analyze behavioral patterns in transaction data"""
        
        # Get user transaction history
        history = await self._get_user_transaction_history(user_id)
        
        # Calculate risk factors
        amount_risk = self._calculate_amount_risk(transaction_data, history)
        velocity_risk = self._calculate_velocity_risk(user_id, transaction_data)
        frequency_risk = self._calculate_frequency_risk(user_id)
        pattern_risk = self._calculate_pattern_risk(transaction_data, history)
        
        device_risk = self._calculate_device_risk(transaction_data)
        location_risk = self._calculate_location_risk(transaction_data, history)
        time_risk = self._calculate_time_risk(transaction_data, history)
        merchant_risk = self._calculate_merchant_risk(transaction_data)
        
        ip_risk = await self._calculate_ip_reputation_risk(transaction_data.get('ip_address'))
        network_risk = await self._calculate_social_network_risk(user_id)
        fraud_network_risk = await self._calculate_fraud_network_risk(transaction_data)
        
        account_risk = self._calculate_account_age_risk(user_id)
        credit_risk = await self._calculate_credit_risk(user_id)
        identity_risk = await self._calculate_identity_risk(user_id)
        kyc_risk = await self._calculate_kyc_risk(user_id)
        
        return RiskFactors(
            amount_risk=amount_risk,
            velocity_risk=velocity_risk,
            frequency_risk=frequency_risk,
            pattern_risk=pattern_risk,
            device_risk=device_risk,
            location_risk=location_risk,
            time_risk=time_risk,
            merchant_risk=merchant_risk,
            ip_reputation_risk=ip_risk,
            social_network_risk=network_risk,
            fraud_network_risk=fraud_network_risk,
            account_age_risk=account_risk,
            credit_risk=credit_risk,
            identity_risk=identity_risk,
            kyc_risk=kyc_risk
        )
    
    def _calculate_amount_risk(self, transaction: Dict, history: List[Dict]) -> float:
        """Calculate risk based on transaction amount patterns"""
        if not history:
            return 0.5  # Medium risk for new accounts
        
        current_amount = transaction.get('amount', 0)
        historical_amounts = [tx.get('amount', 0) for tx in history]
        
        if not historical_amounts:
            return 0.5
        
        # Statistical analysis
        mean_amount = np.mean(historical_amounts)
        std_amount = np.std(historical_amounts)
        max_amount = max(historical_amounts)
        
        # Z-score analysis
        if std_amount > 0:
            z_score = abs(current_amount - mean_amount) / std_amount
            amount_anomaly_score = min(z_score / 3.0, 1.0)  # Normalize to 0-1
        else:
            amount_anomaly_score = 0.0
        
        # Large amount compared to history
        if current_amount > max_amount * 2:
            large_amount_score = 0.8
        elif current_amount > max_amount * 1.5:
            large_amount_score = 0.6
        else:
            large_amount_score = 0.0
        
        # Round number bias (suspicious round amounts)
        round_number_score = 0.3 if current_amount % 100 == 0 and current_amount >= 1000 else 0.0
        
        return min((amount_anomaly_score + large_amount_score + round_number_score) / 3, 1.0)
    
    async def _calculate_velocity_risk(self, user_id: str, transaction: Dict) -> float:
        """Calculate transaction velocity risk"""
        try:
            # Get recent transactions count
            now = datetime.now()
            
            # Last hour
            hour_key = f"velocity:{user_id}:hour:{now.hour}"
            hour_count = await asyncio.to_thread(self.redis_client.get, hour_key)
            hour_count = int(hour_count) if hour_count else 0
            
            # Last 24 hours
            day_key = f"velocity:{user_id}:day:{now.date()}"
            day_count = await asyncio.to_thread(self.redis_client.get, day_key)
            day_count = int(day_count) if day_count else 0
            
            # Velocity risk calculation
            hour_risk = min(hour_count / 10.0, 1.0)  # Risk increases with >10 tx/hour
            day_risk = min(day_count / 100.0, 1.0)   # Risk increases with >100 tx/day
            
            return max(hour_risk, day_risk * 0.7)  # Weight hourly velocity more
            
        except Exception as e:
            logger.error(f"Error calculating velocity risk: {e}")
            return 0.5  # Default moderate risk
    
    def _calculate_frequency_risk(self, user_id: str) -> float:
        """Calculate transaction frequency anomaly risk"""
        try:
            # Get frequency pattern from last 30 days
            frequency_key = f"frequency:{user_id}:30d"
            frequency_data = self.redis_client.get(frequency_key)
            
            if not frequency_data:
                return 0.3  # Low risk for new patterns
            
            frequency_pattern = json.loads(frequency_data)
            
            # Analyze frequency consistency
            daily_counts = frequency_pattern.get('daily_counts', [])
            if len(daily_counts) < 7:
                return 0.3
            
            # Calculate coefficient of variation
            mean_freq = np.mean(daily_counts)
            std_freq = np.std(daily_counts)
            
            if mean_freq > 0:
                cv = std_freq / mean_freq
                return min(cv / 2.0, 1.0)  # High variation = higher risk
            
            return 0.2
            
        except Exception as e:
            logger.error(f"Error calculating frequency risk: {e}")
            return 0.3
    
    def _calculate_pattern_risk(self, transaction: Dict, history: List[Dict]) -> float:
        """Calculate risk based on transaction pattern analysis"""
        if len(history) < 10:
            return 0.2  # Low risk for insufficient data
        
        try:
            # Extract features for pattern analysis
            features = []
            for tx in history[-50:]:  # Last 50 transactions
                features.append([
                    tx.get('amount', 0),
                    tx.get('hour', 12),  # Hour of day
                    tx.get('day_of_week', 1),
                    hash(tx.get('merchant_category', '')) % 100,
                    tx.get('location_risk', 0.5)
                ])
            
            features = np.array(features)
            
            # Current transaction features
            current_features = np.array([[
                transaction.get('amount', 0),
                transaction.get('hour', 12),
                transaction.get('day_of_week', 1),
                hash(transaction.get('merchant_category', '')) % 100,
                transaction.get('location_risk', 0.5)
            ]])
            
            # Fit anomaly detector on historical data
            if len(features) >= 10:
                self.anomaly_detector.fit(features)
                anomaly_score = self.anomaly_detector.decision_function(current_features)[0]
                # Convert to risk score (lower anomaly score = higher risk)
                pattern_risk = max(0, (0.5 - anomaly_score) / 0.5)
                return min(pattern_risk, 1.0)
            
            return 0.3
            
        except Exception as e:
            logger.error(f"Error calculating pattern risk: {e}")
            return 0.3
    
    def _calculate_device_risk(self, transaction: Dict) -> float:
        """Calculate device-based risk factors"""
        device_fingerprint = transaction.get('device_fingerprint', '')
        
        # Device consistency risk
        if not device_fingerprint:
            return 0.8  # High risk for missing device info
        
        # Check device reputation (simplified)
        device_risk_score = 0.0
        
        # New device risk (would check against user's device history)
        if device_fingerprint.startswith('new_'):
            device_risk_score += 0.4
        
        # Suspicious device patterns
        if 'emulator' in device_fingerprint.lower():
            device_risk_score += 0.6
        
        if 'proxy' in device_fingerprint.lower():
            device_risk_score += 0.5
        
        return min(device_risk_score, 1.0)
    
    def _calculate_location_risk(self, transaction: Dict, history: List[Dict]) -> float:
        """Calculate location-based risk"""
        current_lat = transaction.get('location_lat')
        current_lon = transaction.get('location_lon')
        
        if current_lat is None or current_lon is None:
            return 0.6  # Medium-high risk for missing location
        
        # Get historical locations
        historical_locations = [
            (tx.get('location_lat'), tx.get('location_lon'))
            for tx in history
            if tx.get('location_lat') is not None and tx.get('location_lon') is not None
        ]
        
        if not historical_locations:
            return 0.4  # Medium risk for new location pattern
        
        # Calculate minimum distance to historical locations
        min_distance = float('inf')
        for hist_lat, hist_lon in historical_locations:
            distance = self._calculate_distance(current_lat, current_lon, hist_lat, hist_lon)
            min_distance = min(min_distance, distance)
        
        # Distance-based risk (risk increases with distance from usual locations)
        if min_distance > 1000:  # >1000km
            return 0.9
        elif min_distance > 500:  # >500km
            return 0.7
        elif min_distance > 100:  # >100km
            return 0.5
        else:
            return 0.1
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in kilometers"""
        from math import radians, sin, cos, sqrt, atan2
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def _calculate_time_risk(self, transaction: Dict, history: List[Dict]) -> float:
        """Calculate time-based risk patterns"""
        current_hour = transaction.get('hour', datetime.now().hour)
        current_dow = transaction.get('day_of_week', datetime.now().weekday())
        
        if not history:
            return 0.3
        
        # Analyze historical time patterns
        historical_hours = [tx.get('hour', 12) for tx in history]
        historical_dows = [tx.get('day_of_week', 1) for tx in history]
        
        # Unusual hour risk
        hour_counts = np.bincount(historical_hours, minlength=24)
        hour_freq = hour_counts[current_hour] / len(historical_hours) if historical_hours else 0
        hour_risk = 1.0 - hour_freq if hour_freq < 0.1 else 0.0
        
        # Unusual day of week risk
        dow_counts = np.bincount(historical_dows, minlength=7)
        dow_freq = dow_counts[current_dow] / len(historical_dows) if historical_dows else 0
        dow_risk = 1.0 - dow_freq if dow_freq < 0.05 else 0.0
        
        # Late night / early morning risk (2 AM - 6 AM)
        night_risk = 0.4 if 2 <= current_hour <= 6 else 0.0
        
        return min((hour_risk + dow_risk + night_risk) / 3, 1.0)
    
    def _calculate_merchant_risk(self, transaction: Dict) -> float:
        """Calculate merchant-specific risk factors"""
        merchant_id = transaction.get('merchant_id', '')
        merchant_category = transaction.get('merchant_category', '')
        
        # High-risk merchant categories
        high_risk_categories = {
            'gambling', 'adult', 'cryptocurrency', 'money_transfer',
            'prepaid_cards', 'forex', 'online_gaming'
        }
        
        medium_risk_categories = {
            'electronics', 'jewelry', 'luxury_goods', 'travel',
            'digital_goods', 'subscription_services'
        }
        
        category_risk = 0.0
        if merchant_category.lower() in high_risk_categories:
            category_risk = 0.8
        elif merchant_category.lower() in medium_risk_categories:
            category_risk = 0.5
        else:
            category_risk = 0.2
        
        # New merchant risk (would check against user's merchant history)
        merchant_history_risk = 0.3 if merchant_id.startswith('new_') else 0.1
        
        return min((category_risk + merchant_history_risk) / 2, 1.0)
    
    async def _calculate_ip_reputation_risk(self, ip_address: str) -> float:
        """Calculate IP reputation risk"""
        if not ip_address:
            return 0.7
        
        try:
            # Check against IP reputation service (mock implementation)
            reputation_key = f"ip_reputation:{ip_address}"
            cached_reputation = await asyncio.to_thread(self.redis_client.get, reputation_key)
            
            if cached_reputation:
                reputation_data = json.loads(cached_reputation)
                return float(reputation_data.get('risk_score', 0.5))
            
            # Mock IP reputation lookup
            risk_score = 0.3  # Default low risk
            
            # Check for known bad IP patterns
            if ip_address.startswith('10.') or ip_address.startswith('192.168.'):
                risk_score = 0.1  # Private IPs are lower risk
            elif 'tor' in ip_address or 'proxy' in ip_address:
                risk_score = 0.9  # Tor/proxy usage is high risk
            
            # Cache the result
            reputation_data = {'risk_score': risk_score, 'timestamp': datetime.now().isoformat()}
            await asyncio.to_thread(
                self.redis_client.setex,
                reputation_key, 3600, json.dumps(reputation_data)
            )
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating IP reputation risk: {e}")
            return 0.5
    
    async def _calculate_social_network_risk(self, user_id: str) -> float:
        """Calculate social network based risk"""
        try:
            # Check if user is connected to known fraudulent accounts
            network_key = f"social_network:{user_id}"
            network_data = await asyncio.to_thread(self.redis_client.get, network_key)
            
            if not network_data:
                return 0.2  # Low risk for isolated accounts
            
            network_info = json.loads(network_data)
            fraud_connections = network_info.get('fraud_connections', 0)
            total_connections = network_info.get('total_connections', 1)
            
            # Risk increases with fraud connection ratio
            fraud_ratio = fraud_connections / total_connections
            return min(fraud_ratio * 2, 1.0)  # Scale to 0-1
            
        except Exception as e:
            logger.error(f"Error calculating social network risk: {e}")
            return 0.3
    
    async def _calculate_fraud_network_risk(self, transaction: Dict) -> float:
        """Calculate fraud ring network risk"""
        try:
            # Check if transaction involves known fraud network elements
            device_fp = transaction.get('device_fingerprint', '')
            merchant_id = transaction.get('merchant_id', '')
            
            # Check device fingerprint in fraud networks
            device_risk = 0.0
            if device_fp:
                fraud_device_key = f"fraud_device:{device_fp}"
                is_fraud_device = await asyncio.to_thread(self.redis_client.exists, fraud_device_key)
                device_risk = 0.9 if is_fraud_device else 0.0
            
            # Check merchant in fraud networks
            merchant_risk = 0.0
            if merchant_id:
                fraud_merchant_key = f"fraud_merchant:{merchant_id}"
                is_fraud_merchant = await asyncio.to_thread(self.redis_client.exists, fraud_merchant_key)
                merchant_risk = 0.8 if is_fraud_merchant else 0.0
            
            return max(device_risk, merchant_risk)
            
        except Exception as e:
            logger.error(f"Error calculating fraud network risk: {e}")
            return 0.2
    
    def _calculate_account_age_risk(self, user_id: str) -> float:
        """Calculate account age related risk"""
        # Mock account age calculation
        # In production, this would query the user database
        account_age_days = hash(user_id) % 365 + 30  # Mock: 30-395 days
        
        if account_age_days < 30:
            return 0.8  # High risk for very new accounts
        elif account_age_days < 90:
            return 0.6  # Medium-high risk for new accounts
        elif account_age_days < 365:
            return 0.3  # Medium risk for accounts < 1 year
        else:
            return 0.1  # Low risk for established accounts
    
    async def _calculate_credit_risk(self, user_id: str) -> float:
        """Calculate credit/financial risk"""
        try:
            # Mock credit risk assessment
            # In production, this would integrate with credit bureaus
            credit_key = f"credit_risk:{user_id}"
            credit_data = await asyncio.to_thread(self.redis_client.get, credit_key)
            
            if credit_data:
                credit_info = json.loads(credit_data)
                return float(credit_info.get('risk_score', 0.5))
            
            # Mock credit score based on user_id hash
            mock_credit_score = (hash(user_id) % 800) + 300  # 300-1100 range
            
            if mock_credit_score < 500:
                risk_score = 0.8
            elif mock_credit_score < 650:
                risk_score = 0.6
            elif mock_credit_score < 750:
                risk_score = 0.3
            else:
                risk_score = 0.1
            
            # Cache the mock result
            credit_info = {'risk_score': risk_score, 'credit_score': mock_credit_score}
            await asyncio.to_thread(
                self.redis_client.setex,
                credit_key, 86400, json.dumps(credit_info)  # Cache for 24 hours
            )
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating credit risk: {e}")
            return 0.5
    
    async def _calculate_identity_risk(self, user_id: str) -> float:
        """Calculate identity verification risk"""
        try:
            identity_key = f"identity_risk:{user_id}"
            identity_data = await asyncio.to_thread(self.redis_client.get, identity_key)
            
            if identity_data:
                identity_info = json.loads(identity_data)
                return float(identity_info.get('risk_score', 0.5))
            
            # Mock identity verification status
            verification_score = hash(user_id) % 100
            
            if verification_score < 20:  # 20% unverified
                risk_score = 0.9
            elif verification_score < 40:  # 20% partially verified
                risk_score = 0.6
            else:  # 60% fully verified
                risk_score = 0.2
            
            identity_info = {'risk_score': risk_score, 'verification_level': verification_score}
            await asyncio.to_thread(
                self.redis_client.setex,
                identity_key, 86400, json.dumps(identity_info)
            )
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating identity risk: {e}")
            return 0.5
    
    async def _calculate_kyc_risk(self, user_id: str) -> float:
        """Calculate KYC (Know Your Customer) compliance risk"""
        try:
            kyc_key = f"kyc_risk:{user_id}"
            kyc_data = await asyncio.to_thread(self.redis_client.get, kyc_key)
            
            if kyc_data:
                kyc_info = json.loads(kyc_data)
                return float(kyc_info.get('risk_score', 0.5))
            
            # Mock KYC status
            kyc_score = hash(user_id + 'kyc') % 100
            
            if kyc_score < 10:  # 10% no KYC
                risk_score = 0.95
            elif kyc_score < 30:  # 20% basic KYC
                risk_score = 0.7
            elif kyc_score < 70:  # 40% enhanced KYC
                risk_score = 0.4
            else:  # 30% full KYC
                risk_score = 0.1
            
            kyc_info = {'risk_score': risk_score, 'kyc_level': kyc_score}
            await asyncio.to_thread(
                self.redis_client.setex,
                kyc_key, 86400, json.dumps(kyc_info)
            )
            
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating KYC risk: {e}")
            return 0.5
    
    async def _get_user_transaction_history(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get user's transaction history"""
        try:
            history_key = f"transaction_history:{user_id}"
            history_data = await asyncio.to_thread(self.redis_client.get, history_key)
            
            if history_data:
                return json.loads(history_data)[:limit]
            
            # Mock transaction history
            mock_history = []
            for i in range(min(limit, 50)):  # Generate up to 50 mock transactions
                mock_tx = {
                    'amount': np.random.lognormal(5, 1),  # Log-normal distribution for amounts
                    'hour': np.random.randint(6, 23),     # Business hours bias
                    'day_of_week': np.random.choice(range(7), p=[0.1, 0.15, 0.15, 0.15, 0.15, 0.2, 0.1]),  # Weekday bias
                    'merchant_category': np.random.choice(['retail', 'restaurant', 'gas', 'grocery', 'online']),
                    'location_risk': np.random.beta(2, 5),  # Low location risk bias
                    'merchant_id': f"merchant_{np.random.randint(1, 1000)}"
                }
                mock_history.append(mock_tx)
            
            # Cache mock history
            await asyncio.to_thread(
                self.redis_client.setex,
                history_key, 3600, json.dumps(mock_history)
            )
            
            return mock_history
            
        except Exception as e:
            logger.error(f"Error getting transaction history: {e}")
            return []

class RiskScoringEngine:
    """
    Main risk scoring engine that combines all risk factors
    """
    
    def __init__(self):
        self.behavioral_analyzer = BehavioralRiskAnalyzer()
        self.model_version = "2.0.0"
        
        # Risk factor weights (tuned based on historical performance)
        self.risk_weights = {
            # Transaction risks
            'amount_risk': 0.15,
            'velocity_risk': 0.12,
            'frequency_risk': 0.08,
            'pattern_risk': 0.10,
            
            # Behavioral risks
            'device_risk': 0.10,
            'location_risk': 0.08,
            'time_risk': 0.05,
            'merchant_risk': 0.07,
            
            # Network risks
            'ip_reputation_risk': 0.06,
            'social_network_risk': 0.04,
            'fraud_network_risk': 0.08,
            
            # Account risks
            'account_age_risk': 0.03,
            'credit_risk': 0.02,
            'identity_risk': 0.01,
            'kyc_risk': 0.01
        }
    
    async def calculate_risk_score(self, user_id: str, transaction_id: str, transaction_data: Dict) -> RiskScore:
        """
        Calculate comprehensive risk score for a transaction
        """
        start_time = datetime.now()
        
        # Get risk factors
        risk_factors = await self.behavioral_analyzer.analyze_transaction_behavior(
            user_id, transaction_data
        )
        
        # Calculate weighted overall risk score
        overall_risk = 0.0
        risk_factor_dict = asdict(risk_factors)
        
        for factor_name, factor_value in risk_factor_dict.items():
            weight = self.risk_weights.get(factor_name, 0.0)
            overall_risk += factor_value * weight
        
        # Scale to 0-100
        overall_risk_score = min(overall_risk * 100, 100.0)
        
        # Determine risk category
        risk_category = self._get_risk_category(overall_risk_score)
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence(risk_factors)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(overall_risk_score, risk_category)
        
        # Generate explanation
        explanation = self._generate_explanation(risk_factors, overall_risk_score)
        
        return RiskScore(
            user_id=user_id,
            transaction_id=transaction_id,
            overall_risk_score=overall_risk_score,
            risk_category=risk_category,
            risk_factors=risk_factors,
            confidence_level=confidence_level,
            recommendation=recommendation,
            explanation=explanation,
            timestamp=start_time,
            model_version=self.model_version
        )
    
    def _get_risk_category(self, risk_score: float) -> str:
        """Categorize risk based on score"""
        if risk_score >= 80:
            return "CRITICAL"
        elif risk_score >= 60:
            return "HIGH"
        elif risk_score >= 40:
            return "MEDIUM"
        elif risk_score >= 20:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _calculate_confidence(self, risk_factors: RiskFactors) -> float:
        """Calculate confidence in the risk assessment"""
        factor_values = list(asdict(risk_factors).values())
        
        # Confidence is higher when factors are consistent
        # (either mostly high or mostly low risk)
        mean_risk = np.mean(factor_values)
        variance = np.var(factor_values)
        
        # Lower variance = higher confidence
        base_confidence = 1.0 - (variance / 0.25)  # Normalize variance
        
        # Adjust for extreme values
        if mean_risk > 0.8 or mean_risk < 0.2:
            base_confidence += 0.1  # More confident in extreme cases
        
        return min(max(base_confidence, 0.5), 1.0)  # Clamp to 0.5-1.0
    
    def _generate_recommendation(self, risk_score: float, risk_category: str) -> str:
        """Generate action recommendation based on risk"""
        if risk_category == "CRITICAL":
            return "BLOCK_TRANSACTION_IMMEDIATELY"
        elif risk_category == "HIGH":
            return "REQUIRE_MANUAL_REVIEW"
        elif risk_category == "MEDIUM":
            return "REQUIRE_ADDITIONAL_AUTHENTICATION"
        elif risk_category == "LOW":
            return "MONITOR_CLOSELY"
        else:
            return "APPROVE_WITH_STANDARD_MONITORING"
    
    def _generate_explanation(self, risk_factors: RiskFactors, overall_score: float) -> str:
        """Generate human-readable explanation of the risk assessment"""
        factor_dict = asdict(risk_factors)
        
        # Find top risk contributors
        top_risks = sorted(
            [(k, v) for k, v in factor_dict.items() if v > 0.3],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        if not top_risks:
            return f"Low overall risk score ({overall_score:.1f}/100). Transaction appears normal."
        
        explanation = f"Risk score: {overall_score:.1f}/100. "
        explanation += "Primary risk factors: "
        
        risk_descriptions = {
            'amount_risk': 'unusual transaction amount',
            'velocity_risk': 'high transaction velocity',
            'device_risk': 'suspicious device characteristics',
            'location_risk': 'unusual location pattern',
            'fraud_network_risk': 'connection to known fraud networks',
            'ip_reputation_risk': 'suspicious IP reputation',
            'pattern_risk': 'anomalous transaction pattern'
        }
        
        risk_explanations = []
        for risk_name, risk_value in top_risks:
            description = risk_descriptions.get(risk_name, risk_name.replace('_', ' '))
            risk_explanations.append(f"{description} ({risk_value:.1%})")
        
        explanation += ", ".join(risk_explanations)
        
        return explanation

# FastAPI application
app = FastAPI(
    title="FraudGuard 360° - Risk Scoring Service",
    description="Advanced risk assessment for fraud prevention",
    version="2.0.0"
)

# Global scoring engine
scoring_engine = RiskScoringEngine()

# Pydantic models for API
class RiskAssessmentRequest(BaseModel):
    user_id: str = Field(..., description="User account identifier")
    transaction_id: str = Field(..., description="Transaction identifier")
    amount: float = Field(..., ge=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category")
    device_fingerprint: Optional[str] = Field(None, description="Device fingerprint")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    location_lat: Optional[float] = Field(None, description="Latitude")
    location_lon: Optional[float] = Field(None, description="Longitude")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class RiskScoreResponse(BaseModel):
    user_id: str
    transaction_id: str
    overall_risk_score: float
    risk_category: str
    confidence_level: float
    recommendation: str
    explanation: str
    processing_time_ms: float
    model_version: str

@app.post("/assess", response_model=RiskScoreResponse)
async def assess_risk(request: RiskAssessmentRequest):
    """Assess risk for a transaction"""
    start_time = datetime.now()
    
    # Convert request to transaction data
    transaction_data = {
        'amount': request.amount,
        'merchant_id': request.merchant_id,
        'merchant_category': request.merchant_category,
        'device_fingerprint': request.device_fingerprint,
        'ip_address': request.ip_address,
        'location_lat': request.location_lat,
        'location_lon': request.location_lon,
        'hour': request.timestamp.hour,
        'day_of_week': request.timestamp.weekday(),
    }
    
    # Calculate risk score
    risk_score = await scoring_engine.calculate_risk_score(
        request.user_id,
        request.transaction_id,
        transaction_data
    )
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return RiskScoreResponse(
        user_id=risk_score.user_id,
        transaction_id=risk_score.transaction_id,
        overall_risk_score=risk_score.overall_risk_score,
        risk_category=risk_score.risk_category,
        confidence_level=risk_score.confidence_level,
        recommendation=risk_score.recommendation,
        explanation=risk_score.explanation,
        processing_time_ms=processing_time,
        model_version=risk_score.model_version
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Risk Scoring Service",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("risk_scoring_service:app", host="0.0.0.0", port=8002, reload=True)