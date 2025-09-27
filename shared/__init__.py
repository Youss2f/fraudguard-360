"""
FraudGuard 360 Shared Package
"""

from .models import (
    CDR, User, Device, Transaction, FraudAlert, 
    NetworkRelationship, FraudPattern, ProcessingResult,
    CallType, FraudType, AlertSeverity
)
from .utils import (
    KAFKA_TOPICS, DATABASE_CONFIG, RISK_THRESHOLDS, PROCESSING_WINDOWS,
    generate_id, hash_string, calculate_risk_score, 
    get_severity_from_score, format_timestamp, parse_timestamp
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "CDR", "User", "Device", "Transaction", "FraudAlert", 
    "NetworkRelationship", "FraudPattern", "ProcessingResult",
    "CallType", "FraudType", "AlertSeverity",
    # Utils
    "KAFKA_TOPICS", "DATABASE_CONFIG", "RISK_THRESHOLDS", "PROCESSING_WINDOWS",
    "generate_id", "hash_string", "calculate_risk_score", 
    "get_severity_from_score", "format_timestamp", "parse_timestamp"
]