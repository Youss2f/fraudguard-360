"""
Shared utilities and constants for FraudGuard 360 system.
"""

import hashlib
import uuid
from datetime import datetime
from typing import Dict, Any

# Kafka Topics
KAFKA_TOPICS = {
    "CDR_TOPIC": "telecom-cdr-topic",
    "ALERTS_TOPIC": "fraud-alerts-topic",
    "PATTERNS_TOPIC": "fraud-patterns-topic",
    "NETWORK_UPDATES": "network-updates-topic"
}

# Database configurations
DATABASE_CONFIG = {
    "NEO4J": {
        "URI": "bolt://localhost:7687",
        "USERNAME": "neo4j",
        "PASSWORD": "password"
    },
    "POSTGRESQL": {
        "HOST": "localhost",
        "PORT": 5432,
        "USERNAME": "postgres",
        "PASSWORD": "password",
        "DATABASE": "fraudguard"
    }
}

# Risk thresholds
RISK_THRESHOLDS = {
    "CRITICAL": 0.9,
    "HIGH": 0.7,
    "MEDIUM": 0.5,
    "LOW": 0.3
}

# Processing windows (in minutes)
PROCESSING_WINDOWS = {
    "SHORT_TERM": 5,
    "MEDIUM_TERM": 30,
    "LONG_TERM": 60
}


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix."""
    unique_id = str(uuid.uuid4())
    return f"{prefix}-{unique_id}" if prefix else unique_id


def hash_string(value: str) -> str:
    """Generate SHA-256 hash of a string."""
    return hashlib.sha256(value.encode()).hexdigest()


def calculate_risk_score(features: Dict[str, Any]) -> float:
    """Calculate basic risk score based on features."""
    # This is a simplified risk calculation
    # In production, this would use the ML model
    score = 0.0
    
    # Example risk factors
    if features.get("call_frequency", 0) > 100:  # High call frequency
        score += 0.3
    
    if features.get("international_calls", 0) > 10:  # Many international calls
        score += 0.2
    
    if features.get("premium_rate_calls", 0) > 5:  # Premium rate calls
        score += 0.4
    
    if features.get("unusual_hours", False):  # Calls at unusual hours
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0


def get_severity_from_score(score: float) -> str:
    """Convert risk score to severity level."""
    if score >= RISK_THRESHOLDS["CRITICAL"]:
        return "critical"
    elif score >= RISK_THRESHOLDS["HIGH"]:
        return "high"
    elif score >= RISK_THRESHOLDS["MEDIUM"]:
        return "medium"
    else:
        return "low"


def format_timestamp(dt: datetime) -> str:
    """Format datetime for consistent representation."""
    return dt.isoformat()


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO format timestamp string."""
    return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))