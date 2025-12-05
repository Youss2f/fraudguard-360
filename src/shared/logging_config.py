"""
FraudGuard-360 Structured Logging
==================================

Production-grade structured logging with JSON output.

Author: FraudGuard-360 Platform Team
License: MIT
"""

import logging
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextvars import ContextVar

import structlog
from structlog.types import EventDict, Processor

# Context variables for request tracing
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def add_timestamp(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO 8601 timestamp."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_log_level(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Normalize log level."""
    if "level" in event_dict:
        event_dict["level"] = event_dict["level"].upper()
    return event_dict


def add_service_context(service_name: str, service_version: str) -> Processor:
    """Create processor that adds service context."""
    def processor(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
        event_dict["service"] = service_name
        event_dict["version"] = service_version
        return event_dict
    return processor


def add_request_context(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add request context to log entries."""
    request_id = request_id_ctx.get()
    correlation_id = correlation_id_ctx.get()
    
    if request_id:
        event_dict["request_id"] = request_id
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    
    return event_dict


def sanitize_sensitive_data(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Sanitize sensitive data from logs."""
    sensitive_keys = {"password", "secret", "token", "api_key", "authorization", "credential"}
    
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: "***REDACTED***" if any(s in k.lower() for s in sensitive_keys) else _sanitize(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [_sanitize(item) for item in obj]
        return obj
    
    return _sanitize(event_dict)


def format_exception(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Format exception information."""
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        if isinstance(exc_info, BaseException):
            event_dict["exception"] = {
                "type": type(exc_info).__name__,
                "message": str(exc_info),
                "traceback": traceback.format_exception(type(exc_info), exc_info, exc_info.__traceback__)
            }
        elif exc_info is True:
            event_dict["exception"] = {"traceback": traceback.format_exc()}
    return event_dict


class JSONRenderer:
    """Custom JSON renderer for structlog."""
    
    def __call__(self, logger: logging.Logger, method_name: str, event_dict: EventDict) -> str:
        def _serialize(obj: Any) -> Any:
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Exception):
                return str(obj)
            elif hasattr(obj, "__dict__"):
                return str(obj)
            return obj
        
        serializable = {k: _serialize(v) for k, v in event_dict.items()}
        return json.dumps(serializable, default=str, ensure_ascii=False)


def configure_logging(
    service_name: str,
    service_version: str,
    log_level: str = "INFO",
    json_output: bool = True
) -> structlog.BoundLogger:
    """Configure structured logging."""
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        add_timestamp,
        add_log_level,
        add_service_context(service_name, service_version),
        add_request_context,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        format_exception,
        sanitize_sensitive_data,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_output:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)


class LogContext:
    """Context manager for request-specific logging context."""
    
    def __init__(self, request_id: Optional[str] = None, correlation_id: Optional[str] = None, **extra: Any):
        self.request_id = request_id
        self.correlation_id = correlation_id
        self.extra = extra
        self._tokens: Dict[str, Any] = {}
    
    def __enter__(self) -> "LogContext":
        if self.request_id:
            self._tokens["request_id"] = request_id_ctx.set(self.request_id)
        if self.correlation_id:
            self._tokens["correlation_id"] = correlation_id_ctx.set(self.correlation_id)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**self.extra)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, token in self._tokens.items():
            if key == "request_id":
                request_id_ctx.reset(token)
            elif key == "correlation_id":
                correlation_id_ctx.reset(token)
        structlog.contextvars.clear_contextvars()
        return False
