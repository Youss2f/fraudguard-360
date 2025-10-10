"""
FraudGuard 360° - Core ML Service API
FastAPI service for real-time fraud detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta
import json
import os
from contextlib import asynccontextmanager

from fraud_detection_engine import (
    RealTimeFraudDetector, 
    TransactionFeatures, 
    FraudPrediction,
    DEFAULT_MODEL_CONFIG
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global detector instance
detector: Optional[RealTimeFraudDetector] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global detector
    
    # Startup
    logger.info("Initializing FraudGuard 360° ML Service...")
    detector = RealTimeFraudDetector(DEFAULT_MODEL_CONFIG)
    await detector.initialize_models()
    logger.info("ML Service initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML Service...")

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard 360° - Core ML Service",
    description="Enterprise-grade real-time fraud detection with advanced machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class TransactionRequest(BaseModel):
    """Transaction data for fraud analysis"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., ge=0, description="Transaction amount")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category code")
    user_id: str = Field(..., description="User account identifier")
    device_fingerprint: str = Field(..., description="Device fingerprint")
    ip_address: str = Field(..., description="Client IP address")
    location_lat: Optional[float] = Field(None, description="Transaction latitude")
    location_lon: Optional[float] = Field(None, description="Transaction longitude")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    
    @validator('amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        if v > 1000000:  # $1M limit
            raise ValueError('Amount exceeds maximum limit')
        return v

class BatchTransactionRequest(BaseModel):
    """Batch transaction analysis request"""
    transactions: List[TransactionRequest] = Field(..., max_items=1000)
    priority: str = Field(default="normal", regex="^(low|normal|high|critical)$")

class FraudResponse(BaseModel):
    """Fraud analysis response"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: str
    confidence_score: float = Field(..., ge=0, le=1)
    explanation: Dict[str, float]
    model_version: str
    processing_time_ms: float
    recommendations: List[str]
    alert_triggered: bool

class BatchFraudResponse(BaseModel):
    """Batch fraud analysis response"""
    results: List[FraudResponse]
    summary: Dict[str, Any]
    processing_time_ms: float
    total_transactions: int
    flagged_transactions: int

class ModelStatusResponse(BaseModel):
    """ML model status information"""
    model_version: str
    status: str
    last_updated: datetime
    accuracy_metrics: Dict[str, float]
    feature_importance: Dict[str, float]

class HealthResponse(BaseModel):
    """Service health status"""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    memory_usage_mb: float
    active_connections: int

# Service startup time for uptime calculation
startup_time = datetime.now()

# Helper functions
async def get_detector() -> RealTimeFraudDetector:
    """Get the fraud detector instance"""
    if detector is None:
        raise HTTPException(status_code=503, detail="ML service not initialized")
    return detector

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)"""
    # In production, implement proper JWT validation
    if credentials.credentials != "fraudguard-api-key-2025":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

async def convert_to_transaction_features(request: TransactionRequest) -> TransactionFeatures:
    """Convert API request to TransactionFeatures"""
    
    # Calculate derived features (would typically come from feature store)
    current_time = request.timestamp or datetime.now()
    
    # Mock feature calculations (in production, these would be real-time lookups)
    location_risk_score = 0.3 if request.location_lat else 0.5
    account_age_days = 365  # Mock: would lookup from user database
    avg_transaction_amount_30d = 150.0  # Mock: calculated from transaction history
    transaction_count_24h = 5  # Mock: real-time counter
    velocity_score = 25.0  # Mock: transaction velocity calculation
    ip_reputation_score = 0.8  # Mock: IP reputation lookup
    merchant_risk_score = 0.4  # Mock: merchant risk assessment
    time_since_last_transaction = 3600  # Mock: seconds since last transaction
    
    return TransactionFeatures(
        amount=request.amount,
        merchant_category=request.merchant_category,
        transaction_time=current_time,
        location_risk_score=location_risk_score,
        account_age_days=account_age_days,
        avg_transaction_amount_30d=avg_transaction_amount_30d,
        transaction_count_24h=transaction_count_24h,
        velocity_score=velocity_score,
        device_fingerprint=request.device_fingerprint,
        ip_reputation_score=ip_reputation_score,
        merchant_risk_score=merchant_risk_score,
        time_since_last_transaction=time_since_last_transaction
    )

def generate_recommendations(prediction: FraudPrediction) -> List[str]:
    """Generate action recommendations based on fraud prediction"""
    recommendations = []
    
    if prediction.fraud_probability >= 0.8:
        recommendations.extend([
            "BLOCK_TRANSACTION",
            "FREEZE_ACCOUNT",
            "MANUAL_REVIEW_REQUIRED",
            "CONTACT_CARDHOLDER"
        ])
    elif prediction.fraud_probability >= 0.6:
        recommendations.extend([
            "REQUIRE_3DS_AUTHENTICATION",
            "MANUAL_REVIEW_RECOMMENDED",
            "MONITOR_ACCOUNT"
        ])
    elif prediction.fraud_probability >= 0.4:
        recommendations.extend([
            "ADDITIONAL_VERIFICATION",
            "MONITOR_PATTERNS"
        ])
    else:
        recommendations.append("APPROVE_TRANSACTION")
    
    return recommendations

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "FraudGuard 360° Core ML Service",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import psutil
    
    uptime = (datetime.now() - startup_time).total_seconds()
    memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    return HealthResponse(
        status="healthy" if detector else "degraded",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime,
        memory_usage_mb=memory_usage,
        active_connections=1  # Simplified
    )

@app.post("/analyze", response_model=FraudResponse)
async def analyze_transaction(
    request: TransactionRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_token),
    fraud_detector: RealTimeFraudDetector = Depends(get_detector)
):
    """
    Analyze a single transaction for fraud
    
    Returns real-time fraud assessment with explanations and recommendations
    """
    try:
        # Convert request to features
        features = await convert_to_transaction_features(request)
        
        # Get fraud prediction
        prediction = await fraud_detector.predict_fraud(features)
        
        # Generate recommendations
        recommendations = generate_recommendations(prediction)
        
        # Determine if alert should be triggered
        alert_triggered = prediction.fraud_probability >= 0.6
        
        # Log high-risk transactions
        if alert_triggered:
            logger.warning(f"High-risk transaction detected: {request.transaction_id} "
                         f"(probability: {prediction.fraud_probability:.3f})")
        
        # Background task for analytics
        background_tasks.add_task(
            log_transaction_analytics, 
            request.transaction_id, 
            prediction.fraud_probability
        )
        
        return FraudResponse(
            transaction_id=request.transaction_id,
            fraud_probability=prediction.fraud_probability,
            risk_level=prediction.risk_level,
            confidence_score=prediction.confidence_score,
            explanation=prediction.explanation,
            model_version=prediction.model_version,
            processing_time_ms=prediction.processing_time_ms,
            recommendations=recommendations,
            alert_triggered=alert_triggered
        )
        
    except Exception as e:
        logger.error(f"Error analyzing transaction {request.transaction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch", response_model=BatchFraudResponse)
async def analyze_batch_transactions(
    request: BatchTransactionRequest,
    _: str = Depends(verify_token),
    fraud_detector: RealTimeFraudDetector = Depends(get_detector)
):
    """
    Analyze multiple transactions in batch
    
    Optimized for high-throughput fraud detection
    """
    try:
        start_time = datetime.now()
        
        # Convert all requests to features
        feature_tasks = [
            convert_to_transaction_features(tx_request) 
            for tx_request in request.transactions
        ]
        features_list = await asyncio.gather(*feature_tasks)
        
        # Batch prediction
        predictions = await fraud_detector.batch_predict(features_list)
        
        # Process results
        results = []
        flagged_count = 0
        
        for i, prediction in enumerate(predictions):
            tx_request = request.transactions[i]
            recommendations = generate_recommendations(prediction)
            alert_triggered = prediction.fraud_probability >= 0.6
            
            if alert_triggered:
                flagged_count += 1
            
            results.append(FraudResponse(
                transaction_id=tx_request.transaction_id,
                fraud_probability=prediction.fraud_probability,
                risk_level=prediction.risk_level,
                confidence_score=prediction.confidence_score,
                explanation=prediction.explanation,
                model_version=prediction.model_version,
                processing_time_ms=prediction.processing_time_ms,
                recommendations=recommendations,
                alert_triggered=alert_triggered
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Summary statistics
        fraud_probabilities = [r.fraud_probability for r in results]
        summary = {
            "avg_fraud_probability": sum(fraud_probabilities) / len(fraud_probabilities),
            "max_fraud_probability": max(fraud_probabilities),
            "min_fraud_probability": min(fraud_probabilities),
            "flagged_percentage": (flagged_count / len(results)) * 100,
            "risk_distribution": {
                level: len([r for r in results if r.risk_level == level])
                for level in ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            }
        }
        
        return BatchFraudResponse(
            results=results,
            summary=summary,
            processing_time_ms=processing_time,
            total_transactions=len(results),
            flagged_transactions=flagged_count
        )
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status(
    _: str = Depends(verify_token),
    fraud_detector: RealTimeFraudDetector = Depends(get_detector)
):
    """Get ML model status and performance metrics"""
    
    # Mock metrics (in production, these would be real metrics)
    accuracy_metrics = {
        "accuracy": 0.94,
        "precision": 0.89,
        "recall": 0.92,
        "f1_score": 0.90,
        "auc_roc": 0.96
    }
    
    feature_importance = {
        "transaction_amount": 0.25,
        "velocity_score": 0.18,
        "location_risk": 0.15,
        "merchant_risk": 0.12,
        "device_fingerprint": 0.10,
        "ip_reputation": 0.08,
        "time_features": 0.07,
        "account_age": 0.05
    }
    
    return ModelStatusResponse(
        model_version=fraud_detector.model_version,
        status="active",
        last_updated=datetime.now() - timedelta(hours=2),  # Mock
        accuracy_metrics=accuracy_metrics,
        feature_importance=feature_importance
    )

@app.post("/model/retrain")
async def trigger_model_retraining(
    background_tasks: BackgroundTasks,
    _: str = Depends(verify_token)
):
    """Trigger model retraining (background process)"""
    
    background_tasks.add_task(retrain_models_background)
    
    return {
        "message": "Model retraining initiated",
        "status": "accepted",
        "estimated_duration_minutes": 30
    }

# Background tasks
async def log_transaction_analytics(transaction_id: str, fraud_probability: float):
    """Log transaction for analytics and monitoring"""
    try:
        # In production, this would send to analytics pipeline
        logger.info(f"Analytics: Transaction {transaction_id} - "
                   f"Fraud probability: {fraud_probability:.3f}")
        
        # Could send to Kafka, InfluxDB, etc.
        await asyncio.sleep(0.1)  # Simulate processing
        
    except Exception as e:
        logger.error(f"Failed to log analytics for {transaction_id}: {e}")

async def retrain_models_background():
    """Background task for model retraining"""
    try:
        logger.info("Starting model retraining...")
        
        # Simulate retraining process
        await asyncio.sleep(5)  # In production, this would be actual retraining
        
        logger.info("Model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "ml_service_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )