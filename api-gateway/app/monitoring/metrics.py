# ==============================================================================
# FraudGuard 360 - API Gateway Metrics
# Prometheus metrics for API Gateway monitoring
# ==============================================================================

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time
from functools import wraps
from typing import Callable

# API Request Metrics
api_requests_total = Counter(
    'fraudguard_api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

api_request_duration_seconds = Histogram(
    'fraudguard_api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Fraud Detection Metrics
fraud_detections_total = Counter(
    'fraudguard_fraud_detections_total',
    'Total number of fraud detections',
    ['detection_type', 'severity']
)

fraud_detection_accuracy = Gauge(
    'fraudguard_fraud_detection_accuracy',
    'Current fraud detection accuracy percentage'
)

fraud_detection_latency_seconds = Histogram(
    'fraudguard_fraud_detection_latency_seconds',
    'Fraud detection processing latency',
    ['detection_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# CDR Processing Metrics
cdr_processed_total = Counter(
    'fraudguard_cdr_processed_total',
    'Total number of CDRs processed',
    ['status']
)

cdr_processing_duration_seconds = Histogram(
    'fraudguard_cdr_processing_duration_seconds',
    'CDR processing duration in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Alert Metrics
alerts_generated_total = Counter(
    'fraudguard_alerts_generated_total',
    'Total number of alerts generated',
    ['alert_type', 'severity']
)

active_alerts_count = Gauge(
    'fraudguard_active_alerts_count',
    'Number of currently active alerts',
    ['alert_type']
)

# Network Analysis Metrics
network_nodes_analyzed = Counter(
    'fraudguard_network_nodes_analyzed_total',
    'Total number of network nodes analyzed'
)

suspicious_patterns_detected = Counter(
    'fraudguard_suspicious_patterns_detected_total',
    'Total number of suspicious patterns detected',
    ['pattern_type']
)

# Database Metrics
database_operations_total = Counter(
    'fraudguard_database_operations_total',
    'Total number of database operations',
    ['operation', 'database', 'status']
)

database_connection_pool_size = Gauge(
    'fraudguard_database_connection_pool_size',
    'Current database connection pool size',
    ['database']
)

# Cache Metrics
cache_operations_total = Counter(
    'fraudguard_cache_operations_total',
    'Total number of cache operations',
    ['operation', 'result']
)

cache_hit_ratio = Gauge(
    'fraudguard_cache_hit_ratio',
    'Cache hit ratio percentage'
)

# ML Model Metrics
model_inference_total = Counter(
    'fraudguard_model_inference_total',
    'Total number of model inferences',
    ['model_name', 'status']
)

model_inference_duration_seconds = Histogram(
    'fraudguard_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

model_accuracy_score = Gauge(
    'fraudguard_model_accuracy_score',
    'Current model accuracy score',
    ['model_name']
)

# System Health Metrics
system_health_status = Gauge(
    'fraudguard_system_health_status',
    'System health status (1=healthy, 0=unhealthy)',
    ['component']
)

external_service_status = Gauge(
    'fraudguard_external_service_status',
    'External service availability (1=available, 0=unavailable)',
    ['service']
)

# Business Metrics
daily_transaction_volume = Gauge(
    'fraudguard_daily_transaction_volume',
    'Daily transaction volume processed'
)

fraud_loss_prevented_amount = Counter(
    'fraudguard_fraud_loss_prevented_amount',
    'Total amount of fraud loss prevented in currency units'
)

false_positive_rate = Gauge(
    'fraudguard_false_positive_rate',
    'Current false positive rate percentage'
)

# Decorator for timing API requests
def monitor_api_request(endpoint: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            method = "GET"  # Default, should be extracted from request
            start_time = time.time()
            status_code = 200
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                duration = time.time() - start_time
                api_requests_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=str(status_code)
                ).inc()
                api_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator

# Decorator for timing fraud detection
def monitor_fraud_detection(detection_type: str):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record successful detection
                if result and 'fraud_detected' in result:
                    severity = result.get('severity', 'unknown')
                    fraud_detections_total.labels(
                        detection_type=detection_type,
                        severity=severity
                    ).inc()
                
                return result
            finally:
                duration = time.time() - start_time
                fraud_detection_latency_seconds.labels(
                    detection_type=detection_type
                ).observe(duration)
        
        return wrapper
    return decorator

# Health check metrics updater
def update_health_metrics(component: str, is_healthy: bool):
    """Update health status for a component"""
    system_health_status.labels(component=component).set(1 if is_healthy else 0)

def update_external_service_status(service: str, is_available: bool):
    """Update external service availability"""
    external_service_status.labels(service=service).set(1 if is_available else 0)

# Business metrics updaters
def record_fraud_detection(detection_type: str, severity: str, amount_saved: float = 0):
    """Record a fraud detection event"""
    fraud_detections_total.labels(
        detection_type=detection_type,
        severity=severity
    ).inc()
    
    if amount_saved > 0:
        fraud_loss_prevented_amount.inc(amount_saved)

def update_model_accuracy(model_name: str, accuracy: float):
    """Update model accuracy metrics"""
    model_accuracy_score.labels(model_name=model_name).set(accuracy)
    fraud_detection_accuracy.set(accuracy)

def record_database_operation(operation: str, database: str, success: bool):
    """Record database operation"""
    status = "success" if success else "error"
    database_operations_total.labels(
        operation=operation,
        database=database,
        status=status
    ).inc()

def record_cache_operation(operation: str, hit: bool):
    """Record cache operation"""
    result = "hit" if hit else "miss"
    cache_operations_total.labels(
        operation=operation,
        result=result
    ).inc()

def record_model_inference(model_name: str, duration: float, success: bool):
    """Record model inference metrics"""
    status = "success" if success else "error"
    model_inference_total.labels(
        model_name=model_name,
        status=status
    ).inc()
    
    if success:
        model_inference_duration_seconds.labels(
            model_name=model_name
        ).observe(duration)

def record_cdr_processing(duration: float, success: bool):
    """Record CDR processing metrics"""
    status = "success" if success else "error"
    cdr_processed_total.labels(status=status).inc()
    
    if success:
        cdr_processing_duration_seconds.observe(duration)

def record_alert_generation(alert_type: str, severity: str):
    """Record alert generation"""
    alerts_generated_total.labels(
        alert_type=alert_type,
        severity=severity
    ).inc()

def update_active_alerts_count(alert_type: str, count: int):
    """Update active alerts count"""
    active_alerts_count.labels(alert_type=alert_type).set(count)

def record_network_analysis(nodes_count: int, patterns_found: dict):
    """Record network analysis metrics"""
    network_nodes_analyzed.inc(nodes_count)
    
    for pattern_type, count in patterns_found.items():
        suspicious_patterns_detected.labels(
            pattern_type=pattern_type
        ).inc(count)

# Metrics endpoint
async def get_metrics():
    """Return Prometheus metrics"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )