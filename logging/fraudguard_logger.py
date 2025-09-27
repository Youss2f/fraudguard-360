import logging
import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger


class FraudGuardFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for FraudGuard 360 with fraud-specific fields"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add standard fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['service'] = getattr(record, 'service', 'fraudguard')
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add trace ID if present
        if hasattr(record, 'trace_id'):
            log_record['trace_id'] = record.trace_id
        
        # Add user ID if present
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        
        # Add fraud detection specific fields
        if hasattr(record, 'fraud_detection'):
            log_record['fraud_detection'] = record.fraud_detection
        
        # Add performance metrics
        if hasattr(record, 'performance'):
            log_record['performance'] = record.performance
        
        # Add request context
        if hasattr(record, 'request_context'):
            log_record['request_context'] = record.request_context


class FraudGuardLogger:
    """Enhanced logger for FraudGuard 360 with structured logging and fraud-specific context"""
    
    def __init__(self, name: str, service: str = "fraudguard"):
        self.logger = logging.getLogger(name)
        self.service = service
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup console and file handlers with JSON formatting"""
        if self.logger.handlers:
            return
        
        self.logger.setLevel(logging.INFO)
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler()
        console_formatter = FraudGuardFormatter(
            '%(timestamp)s %(level)s %(service)s %(module)s.%(function)s:%(line)d %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            file_handler = logging.FileHandler('/var/log/fraudguard/app.log')
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
        except (OSError, PermissionError):
            # Fallback to local file if /var/log is not writable
            file_handler = logging.FileHandler('fraudguard.log')
            file_handler.setFormatter(console_formatter)
            self.logger.addHandler(file_handler)
    
    def _log_with_context(self, level: int, message: str, **kwargs) -> None:
        """Log with additional context"""
        extra = {
            'service': self.service,
            **kwargs
        }
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with context"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with context"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with context and stack trace"""
        if error:
            kwargs['error_type'] = error.__class__.__name__
            kwargs['error_message'] = str(error)
            kwargs['stack_trace'] = traceback.format_exc()
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with context"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def fraud_detection(self, message: str, fraud_score: float, fraud_type: str, 
                       severity: str, user_id: Optional[str] = None, 
                       transaction_id: Optional[str] = None, **kwargs) -> None:
        """Log fraud detection events with specialized context"""
        fraud_context = {
            'fraud_detection': {
                'score': fraud_score,
                'type': fraud_type,
                'severity': severity,
                'user_id': user_id,
                'transaction_id': transaction_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        self._log_with_context(logging.WARNING if fraud_score > 0.5 else logging.INFO, 
                              message, **fraud_context, **kwargs)
    
    def api_request(self, method: str, path: str, status_code: int, 
                   duration: float, user_id: Optional[str] = None, 
                   trace_id: Optional[str] = None, **kwargs) -> None:
        """Log API request with context"""
        request_context = {
            'request_context': {
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration_ms': duration * 1000,
                'user_id': user_id,
                'trace_id': trace_id
            }
        }
        self._log_with_context(logging.INFO, f"API request: {method} {path}", 
                              **request_context, **kwargs)
    
    def model_performance(self, model_name: str, accuracy: float, 
                         latency: float, **kwargs) -> None:
        """Log ML model performance metrics"""
        performance_context = {
            'performance': {
                'model_name': model_name,
                'accuracy': accuracy,
                'latency_ms': latency,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        self._log_with_context(logging.INFO, f"Model performance: {model_name}", 
                              **performance_context, **kwargs)
    
    def cdr_processing(self, cdr_type: str, call_duration: float, 
                      call_cost: float, fraud_score: float, **kwargs) -> None:
        """Log CDR processing events"""
        cdr_context = {
            'cdr_processing': {
                'type': cdr_type,
                'duration': call_duration,
                'cost': call_cost,
                'fraud_score': fraud_score,
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        self._log_with_context(logging.INFO, f"CDR {cdr_type} processed", 
                              **cdr_context, **kwargs)


# Global logger instances for different services
api_gateway_logger = FraudGuardLogger("api_gateway", "api-gateway")
ml_service_logger = FraudGuardLogger("ml_service", "ml-service")
flink_jobs_logger = FraudGuardLogger("flink_jobs", "flink-jobs")
general_logger = FraudGuardLogger("fraudguard", "general")


def get_logger(service: str = "fraudguard") -> FraudGuardLogger:
    """Get logger instance for specific service"""
    service_loggers = {
        "api-gateway": api_gateway_logger,
        "ml-service": ml_service_logger,
        "flink-jobs": flink_jobs_logger
    }
    return service_loggers.get(service, general_logger)