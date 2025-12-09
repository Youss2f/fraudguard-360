"""
FraudGuard-360 Database Models - Telecom Fraud Detection
=========================================================

SQLAlchemy models for the FraudGuard-360 Telecom Fraud Detection platform.
Domain: Call Detail Records (CDR) for detecting Wangiri, SIM Box, and other telecom fraud.

Author: FraudGuard-360 Platform Team
License: MIT
"""

import enum
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Enum, 
    Boolean, ForeignKey, Index, JSON, create_engine
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func
import uuid

from shared.config import get_settings

Base = declarative_base()


class CDRStatus(str, enum.Enum):
    """CDR processing status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    FLAGGED = "flagged"
    UNDER_REVIEW = "under_review"


class CallType(str, enum.Enum):
    """Call type enumeration for CDR."""
    VOICE = "voice"
    SMS = "sms"


class FraudType(str, enum.Enum):
    """Fraud type classification."""
    NONE = "none"
    WANGIRI = "wangiri"
    SIMBOX = "simbox"
    IRSF = "irsf"
    PBX_HACKING = "pbx_hacking"


class RiskLevel(str, enum.Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSeverity(str, enum.Enum):
    """Alert severity enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Backward compatibility alias
TransactionStatus = CDRStatus


class Subscriber(Base):
    """Subscriber model for phone numbers (MSISDNs)."""
    __tablename__ = "subscribers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    msisdn = Column(String(20), unique=True, nullable=False, index=True)
    imsi = Column(String(20), nullable=True)
    name = Column(String(255), nullable=True)
    subscriber_type = Column(String(50), default="prepaid")  # prepaid, postpaid, corporate
    risk_score = Column(Float, default=0.0)
    total_calls = Column(Integer, default=0)
    flagged_calls = Column(Integer, default=0)
    country_code = Column(String(5), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_ = Column("metadata", JSON, default=dict)
    
    # Relationships
    cdrs_as_caller = relationship("CDR", back_populates="caller", foreign_keys="CDR.caller_msisdn")
    
    __table_args__ = (
        Index('ix_subscribers_risk_score', 'risk_score'),
    )


# Backward compatibility aliases
User = Subscriber


class CellTower(Base):
    """Cell Tower model for location tracking."""
    __tablename__ = "cell_towers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tower_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    risk_score = Column(Float, default=0.0)
    total_calls = Column(Integer, default=0)
    flagged_calls = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    cdrs = relationship("CDR", back_populates="cell_tower")


# Backward compatibility alias
Merchant = CellTower


class CDR(Base):
    """Call Detail Record - core entity for telecom fraud detection."""
    __tablename__ = "cdrs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cdr_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Call parties
    caller_msisdn = Column(String(20), ForeignKey("subscribers.msisdn"), nullable=False, index=True)
    callee_msisdn = Column(String(20), nullable=False, index=True)
    
    # Call details
    duration = Column(Integer, nullable=False, default=0)  # Seconds
    call_type = Column(Enum(CallType), default=CallType.VOICE)
    cell_tower_id = Column(String(50), ForeignKey("cell_towers.tower_id"), nullable=True, index=True)
    roaming = Column(Boolean, default=False)
    imei = Column(String(20), nullable=True, index=True)
    
    # Fraud detection
    fraud_type = Column(Enum(FraudType), default=FraudType.NONE)
    risk_score = Column(Float, default=0.0)
    risk_level = Column(Enum(RiskLevel), default=RiskLevel.LOW)
    risk_factors = Column(JSON, default=list)
    
    # Status & decisions
    status = Column(Enum(CDRStatus), default=CDRStatus.PENDING)
    decision = Column(String(50), default="pending")  # approve, block, review
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_notes = Column(Text, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Model info
    model_version = Column(String(20), default="1.0.0")
    processing_time_ms = Column(Float, nullable=True)
    
    # Raw data
    raw_data = Column(JSON, default=dict)
    
    # Relationships
    caller = relationship("Subscriber", back_populates="cdrs_as_caller", foreign_keys=[caller_msisdn])
    cell_tower = relationship("CellTower", back_populates="cdrs")
    alerts = relationship("Alert", back_populates="cdr")
    
    __table_args__ = (
        Index('ix_cdrs_timestamp', 'timestamp'),
        Index('ix_cdrs_created_at', 'created_at'),
        Index('ix_cdrs_status', 'status'),
        Index('ix_cdrs_risk_level', 'risk_level'),
        Index('ix_cdrs_risk_score', 'risk_score'),
        Index('ix_cdrs_fraud_type', 'fraud_type'),
        Index('ix_cdrs_imei', 'imei'),
    )


# Backward compatibility alias
Transaction = CDR


class Alert(Base):
    """Alert model for fraud alerts and notifications."""
    __tablename__ = "alerts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    alert_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Link to CDR
    cdr_id = Column(String(50), ForeignKey("cdrs.cdr_id"), nullable=True, index=True)
    
    # Alert details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.INFO)
    category = Column(String(100), nullable=True)  # wangiri_detected, simbox_detected, irsf_attempt
    
    # Status
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(String(100), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Additional data
    metadata_ = Column("metadata", JSON, default=dict)
    
    # Relationships
    cdr = relationship("CDR", back_populates="alerts")
    
    # Backward compatibility
    @property
    def transaction_id(self):
        return self.cdr_id
    
    @property
    def transaction(self):
        return self.cdr
    
    __table_args__ = (
        Index('ix_alerts_severity', 'severity'),
        Index('ix_alerts_is_read', 'is_read'),
        Index('ix_alerts_created_at', 'created_at'),
    )


class RiskLog(Base):
    """Risk assessment log for audit trail."""
    __tablename__ = "risk_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cdr_id = Column(String(50), index=True, nullable=False)
    
    # Scoring details
    initial_score = Column(Float, nullable=True)
    final_score = Column(Float, nullable=False)
    rules_triggered = Column(JSON, default=list)
    
    # Model info
    model_name = Column(String(100), nullable=True)
    model_version = Column(String(20), nullable=True)
    
    # Processing
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GraphNode(Base):
    """Graph node for Neo4j sync - stores node metadata in PostgreSQL."""
    __tablename__ = "graph_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_id = Column(String(100), unique=True, nullable=False, index=True)
    node_type = Column(String(50), nullable=False)  # msisdn, cell_tower, imei
    label = Column(String(255), nullable=True)
    properties = Column(JSON, default=dict)
    risk_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('ix_graph_nodes_type', 'node_type'),
    )


class GraphEdge(Base):
    """Graph edge for Neo4j sync - stores edge metadata in PostgreSQL."""
    __tablename__ = "graph_edges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(String(100), nullable=False, index=True)
    target_id = Column(String(100), nullable=False, index=True)
    edge_type = Column(String(50), nullable=False)  # called, used_tower, used_imei
    weight = Column(Float, default=1.0)
    properties = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('ix_graph_edges_source', 'source_id'),
        Index('ix_graph_edges_target', 'target_id'),
    )


# Database connection utilities
def get_database_url() -> str:
    """Get PostgreSQL database URL from settings."""
    settings = get_settings()
    return settings.database.url


def get_engine(echo: bool = False):
    """Create SQLAlchemy engine."""
    return create_engine(get_database_url(), echo=echo, pool_pre_ping=True)


def get_session_factory():
    """Create session factory."""
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Initialize database - create all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    return engine


def get_db():
    """Dependency for getting database session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Async support (for FastAPI)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker as async_sessionmaker


def get_async_database_url() -> str:
    """Get async PostgreSQL database URL."""
    url = get_database_url()
    return url.replace("postgresql://", "postgresql+asyncpg://")


def get_async_engine():
    """Create async SQLAlchemy engine."""
    return create_async_engine(get_async_database_url(), echo=False, pool_pre_ping=True)


def get_async_session_factory():
    """Create async session factory."""
    engine = get_async_engine()
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def get_async_db():
    """Async dependency for getting database session."""
    AsyncSessionLocal = get_async_session_factory()
    async with AsyncSessionLocal() as session:
        yield session
