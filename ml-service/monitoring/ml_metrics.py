# ==============================================================================
# FraudGuard 360 - ML Service Metrics
# Prometheus metrics for machine learning service
# ==============================================================================

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time
from functools import wraps
import torch
import numpy as np

# Model Performance Metrics
model_predictions_total = Counter(
    'fraudguard_model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'prediction_type']
)

model_inference_duration_seconds = Histogram(
    'fraudguard_model_inference_duration_seconds',
    'Model inference time in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

model_accuracy_score = Gauge(
    'fraudguard_model_accuracy_score',
    'Current model accuracy score',
    ['model_name']
)

model_precision_score = Gauge(
    'fraudguard_model_precision_score',
    'Current model precision score',
    ['model_name']
)

model_recall_score = Gauge(
    'fraudguard_model_recall_score',
    'Current model recall score',
    ['model_name']
)

model_f1_score = Gauge(
    'fraudguard_model_f1_score',
    'Current model F1 score',
    ['model_name']
)

# Fraud Detection Specific Metrics
fraud_cases_detected_total = Counter(
    'fraudguard_fraud_cases_detected_total',
    'Total fraud cases detected',
    ['severity', 'pattern_type']
)

fraud_score_distribution = Histogram(
    'fraudguard_fraud_score_distribution',
    'Distribution of fraud scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

false_positive_rate = Gauge(
    'fraudguard_false_positive_rate',
    'Current false positive rate'
)

false_negative_rate = Gauge(
    'fraudguard_false_negative_rate',
    'Current false negative rate'
)

# Training Metrics
model_training_duration_seconds = Histogram(
    'fraudguard_model_training_duration_seconds',
    'Model training time in seconds',
    ['model_name'],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400, 28800]  # 1min to 8hrs
)

model_training_loss = Gauge(
    'fraudguard_model_training_loss',
    'Current training loss',
    ['model_name', 'epoch']
)

model_validation_loss = Gauge(
    'fraudguard_model_validation_loss',
    'Current validation loss',
    ['model_name', 'epoch']
)

training_samples_processed = Counter(
    'fraudguard_training_samples_processed_total',
    'Total training samples processed',
    ['model_name']
)

# Data Processing Metrics
feature_extraction_duration_seconds = Histogram(
    'fraudguard_feature_extraction_duration_seconds',
    'Feature extraction time in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

graph_construction_duration_seconds = Histogram(
    'fraudguard_graph_construction_duration_seconds',
    'Graph construction time in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

nodes_processed_total = Counter(
    'fraudguard_nodes_processed_total',
    'Total graph nodes processed'
)

edges_processed_total = Counter(
    'fraudguard_edges_processed_total',
    'Total graph edges processed'
)

# Memory and Resource Metrics
gpu_memory_usage_bytes = Gauge(
    'fraudguard_gpu_memory_usage_bytes',
    'GPU memory usage in bytes',
    ['gpu_id']
)

gpu_utilization_percent = Gauge(
    'fraudguard_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)

model_memory_usage_bytes = Gauge(
    'fraudguard_model_memory_usage_bytes',
    'Model memory usage in bytes',
    ['model_name']
)

# API Specific Metrics
api_requests_total = Counter(
    'fraudguard_ml_api_requests_total',
    'Total API requests to ML service',
    ['endpoint', 'method', 'status_code']
)

api_request_duration_seconds = Histogram(
    'fraudguard_ml_api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Business Impact Metrics
potential_fraud_amount = Counter(
    'fraudguard_potential_fraud_amount_total',
    'Total potential fraud amount detected in currency units'
)

fraud_prevention_savings = Counter(
    'fraudguard_fraud_prevention_savings_total',
    'Total savings from fraud prevention in currency units'
)

# Decorator for monitoring model predictions
def monitor_prediction(model_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful prediction
                prediction_type = "fraud" if result.get('fraud_score', 0) > 0.5 else "legitimate"
                model_predictions_total.labels(
                    model_name=model_name,
                    prediction_type=prediction_type
                ).inc()
                
                # Record fraud score distribution
                fraud_score = result.get('fraud_score', 0)
                fraud_score_distribution.observe(fraud_score)
                
                # Record fraud detection if applicable
                if fraud_score > 0.7:
                    severity = result.get('risk_level', 'medium')
                    pattern_type = result.get('detected_patterns', ['unknown'])[0] if result.get('detected_patterns') else 'unknown'
                    fraud_cases_detected_total.labels(
                        severity=severity,
                        pattern_type=pattern_type
                    ).inc()
                    
                    # Record potential fraud amount
                    amount = result.get('transaction_amount', 0)
                    if amount > 0:
                        potential_fraud_amount.inc(amount)
                
                return result
                
            finally:
                duration = time.time() - start_time
                model_inference_duration_seconds.labels(
                    model_name=model_name
                ).observe(duration)
        
        return wrapper
    return decorator

# Decorator for monitoring training
def monitor_training(model_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                model_training_duration_seconds.labels(
                    model_name=model_name
                ).observe(duration)
        
        return wrapper
    return decorator

# Update model performance metrics
def update_model_metrics(model_name: str, accuracy: float, precision: float, recall: float, f1: float):
    """Update model performance metrics"""
    model_accuracy_score.labels(model_name=model_name).set(accuracy)
    model_precision_score.labels(model_name=model_name).set(precision)
    model_recall_score.labels(model_name=model_name).set(recall)
    model_f1_score.labels(model_name=model_name).set(f1)

def update_error_rates(false_positive: float, false_negative: float):
    """Update false positive and negative rates"""
    false_positive_rate.set(false_positive)
    false_negative_rate.set(false_negative)

def record_training_metrics(model_name: str, epoch: int, train_loss: float, val_loss: float, samples: int):
    """Record training metrics for an epoch"""
    model_training_loss.labels(model_name=model_name, epoch=str(epoch)).set(train_loss)
    model_validation_loss.labels(model_name=model_name, epoch=str(epoch)).set(val_loss)
    training_samples_processed.labels(model_name=model_name).inc(samples)

def record_feature_extraction_time(duration: float):
    """Record feature extraction time"""
    feature_extraction_duration_seconds.observe(duration)

def record_graph_construction_time(duration: float, nodes: int, edges: int):
    """Record graph construction metrics"""
    graph_construction_duration_seconds.observe(duration)
    nodes_processed_total.inc(nodes)
    edges_processed_total.inc(edges)

def update_gpu_metrics():
    """Update GPU usage metrics if available"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            # Get GPU memory info
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            
            gpu_memory_usage_bytes.labels(gpu_id=str(i)).set(memory_allocated)
            
            # GPU utilization would require nvidia-ml-py package
            # For now, we'll set a placeholder
            gpu_utilization_percent.labels(gpu_id=str(i)).set(0)

def record_model_memory_usage(model_name: str, model):
    """Record model memory usage"""
    if hasattr(model, 'parameters'):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        model_memory_usage_bytes.labels(model_name=model_name).set(total_size)

def record_api_request(endpoint: str, method: str, status_code: int, duration: float):
    """Record API request metrics"""
    api_requests_total.labels(
        endpoint=endpoint,
        method=method,
        status_code=str(status_code)
    ).inc()
    
    api_request_duration_seconds.labels(endpoint=endpoint).observe(duration)

def record_fraud_prevention_impact(amount_saved: float):
    """Record fraud prevention business impact"""
    fraud_prevention_savings.inc(amount_saved)

# Metrics endpoint
async def get_ml_metrics():
    """Return Prometheus metrics for ML service"""
    # Update GPU metrics before returning
    update_gpu_metrics()
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )