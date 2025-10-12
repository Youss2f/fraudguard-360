"""
Case Management System for FraudGuard 360
Manages fraud investigation cases, workflows, and analyst assignments
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum
import asyncio
import json
import logging
import uuid
import sqlite3
import aiosqlite
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaseStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    UNDER_REVIEW = "under_review"
    CLOSED_RESOLVED = "closed_resolved"
    CLOSED_FALSE_POSITIVE = "closed_false_positive"
    ESCALATED = "escalated"
    ON_HOLD = "on_hold"

class CasePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CaseType(str, Enum):
    FRAUD_INVESTIGATION = "fraud_investigation"
    ACCOUNT_COMPROMISE = "account_compromise"
    NETWORK_ABUSE = "network_abuse"
    FINANCIAL_FRAUD = "financial_fraud"
    IDENTITY_THEFT = "identity_theft"
    SYSTEM_INTRUSION = "system_intrusion"

class ActionType(str, Enum):
    CREATED = "created"
    ASSIGNED = "assigned"
    STATUS_CHANGED = "status_changed"
    EVIDENCE_ADDED = "evidence_added"
    NOTE_ADDED = "note_added"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CONTACT_ATTEMPTED = "contact_attempted"
    ACCOUNT_BLOCKED = "account_blocked"

class CaseAction(BaseModel):
    """Case action/activity model"""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    action_type: ActionType
    performed_by: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Evidence(BaseModel):
    """Evidence model for case investigations"""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    type: str  # "cdr", "network_log", "screenshot", "document", etc.
    title: str
    description: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collected_by: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    hash_value: Optional[str] = None  # File integrity check
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class FraudCase(BaseModel):
    """Main fraud case model"""
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_number: str  # Human-readable case number
    title: str
    description: str
    case_type: CaseType
    status: CaseStatus = CaseStatus.OPEN
    priority: CasePriority = CasePriority.MEDIUM
    
    # Parties involved
    affected_user_id: str
    reported_by: str
    assigned_to: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Financial impact
    estimated_loss: Optional[float] = Field(None, ge=0)
    recovered_amount: Optional[float] = Field(None, ge=0)
    
    # Classification
    tags: List[str] = Field(default_factory=list)
    related_cases: List[str] = Field(default_factory=list)
    
    # Workflow
    sla_hours: int = Field(default=72, ge=1)  # Service Level Agreement
    escalation_level: int = Field(default=0, ge=0, le=3)
    
    # Analysis
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_level: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Resolution
    resolution_summary: Optional[str] = None
    lessons_learned: Optional[str] = None
    preventive_measures: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CaseUpdate(BaseModel):
    """Case update model"""
    status: Optional[CaseStatus] = None
    priority: Optional[CasePriority] = None
    assigned_to: Optional[str] = None
    description: Optional[str] = None
    estimated_loss: Optional[float] = None
    recovered_amount: Optional[float] = None
    tags: Optional[List[str]] = None
    due_date: Optional[datetime] = None
    resolution_summary: Optional[str] = None
    lessons_learned: Optional[str] = None
    preventive_measures: Optional[List[str]] = None

class CaseQuery(BaseModel):
    """Case query filters"""
    case_type: Optional[CaseType] = None
    status: Optional[CaseStatus] = None
    priority: Optional[CasePriority] = None
    assigned_to: Optional[str] = None
    affected_user_id: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = None
    limit: int = Field(default=50, le=200)
    offset: int = Field(default=0, ge=0)

class CaseStatistics(BaseModel):
    """Case management statistics"""
    total_cases: int
    open_cases: int
    in_progress_cases: int
    resolved_cases: int
    overdue_cases: int
    high_priority_cases: int
    avg_resolution_time_hours: float
    cases_by_type: Dict[str, int]
    cases_by_analyst: Dict[str, int]
    monthly_trend: List[Dict[str, Any]]
    sla_compliance_rate: float

class CaseManager:
    """Main case management service"""
    
    def __init__(self, db_path: str = "cases.db"):
        self.db_path = db_path
        self.case_counter = 1  # For generating case numbers
    
    async def initialize_database(self):
        """Initialize SQLite database for case management"""
        async with aiosqlite.connect(self.db_path) as db:
            # Cases table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    case_number TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    case_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    affected_user_id TEXT NOT NULL,
                    reported_by TEXT NOT NULL,
                    assigned_to TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    due_date TEXT,
                    resolved_at TEXT,
                    estimated_loss REAL,
                    recovered_amount REAL,
                    tags TEXT,
                    related_cases TEXT,
                    sla_hours INTEGER DEFAULT 72,
                    escalation_level INTEGER DEFAULT 0,
                    risk_score REAL DEFAULT 0.0,
                    confidence_level REAL DEFAULT 0.0,
                    resolution_summary TEXT,
                    lessons_learned TEXT,
                    preventive_measures TEXT
                )
            """)
            
            # Case actions table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS case_actions (
                    action_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    performed_by TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    description TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id)
                )
            """)
            
            # Evidence table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    file_path TEXT,
                    metadata TEXT,
                    collected_by TEXT NOT NULL,
                    collected_at TEXT NOT NULL,
                    hash_value TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id)
                )
            """)
            
            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_cases_assigned_to ON cases(assigned_to)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_cases_created_at ON cases(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_actions_case_id ON case_actions(case_id)")
            
            await db.commit()
    
    async def create_case(self, case: FraudCase) -> FraudCase:
        """Create a new fraud case"""
        try:
            # Generate case number
            case.case_number = await self._generate_case_number()
            
            # Set due date based on SLA
            if not case.due_date:
                case.due_date = case.created_at + timedelta(hours=case.sla_hours)
            
            # Store in database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO cases (
                        case_id, case_number, title, description, case_type, status, priority,
                        affected_user_id, reported_by, assigned_to, created_at, updated_at,
                        due_date, resolved_at, estimated_loss, recovered_amount, tags,
                        related_cases, sla_hours, escalation_level, risk_score,
                        confidence_level, resolution_summary, lessons_learned, preventive_measures
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    case.case_id, case.case_number, case.title, case.description,
                    case.case_type.value, case.status.value, case.priority.value,
                    case.affected_user_id, case.reported_by, case.assigned_to,
                    case.created_at.isoformat(), case.updated_at.isoformat(),
                    case.due_date.isoformat() if case.due_date else None,
                    case.resolved_at.isoformat() if case.resolved_at else None,
                    case.estimated_loss, case.recovered_amount,
                    json.dumps(case.tags), json.dumps(case.related_cases),
                    case.sla_hours, case.escalation_level, case.risk_score,
                    case.confidence_level, case.resolution_summary,
                    case.lessons_learned, json.dumps(case.preventive_measures)
                ))
                await db.commit()
            
            # Log creation action
            await self.add_case_action(CaseAction(
                case_id=case.case_id,
                action_type=ActionType.CREATED,
                performed_by=case.reported_by,
                description=f"Case {case.case_number} created: {case.title}"
            ))
            
            logger.info(f"Created case {case.case_number}: {case.title}")
            return case
            
        except Exception as e:
            logger.error(f"Failed to create case: {e}")
            raise HTTPException(status_code=500, detail="Failed to create case")
    
    async def update_case(self, case_id: str, update: CaseUpdate, updated_by: str) -> Optional[FraudCase]:
        """Update an existing case"""
        try:
            # Get current case
            case = await self.get_case(case_id)
            if not case:
                return None
            
            # Track changes for logging
            changes = []
            old_status = case.status
            
            # Apply updates
            update_data = update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(case, field):
                    old_value = getattr(case, field)
                    setattr(case, field, value)
                    if old_value != value:
                        changes.append(f"{field}: {old_value} → {value}")
            
            case.updated_at = datetime.now(timezone.utc)
            
            # Set resolved timestamp if status changed to resolved
            if update.status and update.status in [CaseStatus.CLOSED_RESOLVED, CaseStatus.CLOSED_FALSE_POSITIVE]:
                case.resolved_at = datetime.now(timezone.utc)
            
            # Update in database
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE cases SET
                        status = ?, priority = ?, assigned_to = ?, description = ?,
                        updated_at = ?, estimated_loss = ?, recovered_amount = ?,
                        tags = ?, due_date = ?, resolved_at = ?, resolution_summary = ?,
                        lessons_learned = ?, preventive_measures = ?
                    WHERE case_id = ?
                """, (
                    case.status.value, case.priority.value, case.assigned_to,
                    case.description, case.updated_at.isoformat(),
                    case.estimated_loss, case.recovered_amount,
                    json.dumps(case.tags),
                    case.due_date.isoformat() if case.due_date else None,
                    case.resolved_at.isoformat() if case.resolved_at else None,
                    case.resolution_summary, case.lessons_learned,
                    json.dumps(case.preventive_measures), case_id
                ))
                await db.commit()
            
            # Log update actions
            if changes:
                action_type = ActionType.STATUS_CHANGED if old_status != case.status else ActionType.NOTE_ADDED
                await self.add_case_action(CaseAction(
                    case_id=case_id,
                    action_type=action_type,
                    performed_by=updated_by,
                    description=f"Case updated: {', '.join(changes[:3])}{'...' if len(changes) > 3 else ''}",
                    details={"changes": changes}
                ))
            
            logger.info(f"Updated case {case.case_number}")
            return case
            
        except Exception as e:
            logger.error(f"Failed to update case {case_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to update case")
    
    async def get_case(self, case_id: str) -> Optional[FraudCase]:
        """Get a specific case"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return self._row_to_case(row)
        except Exception as e:
            logger.error(f"Failed to get case {case_id}: {e}")
        return None
    
    async def query_cases(self, query: CaseQuery) -> List[FraudCase]:
        """Query cases with filters"""
        try:
            conditions = []
            params = []
            
            if query.case_type:
                conditions.append("case_type = ?")
                params.append(query.case_type.value)
            
            if query.status:
                conditions.append("status = ?")
                params.append(query.status.value)
            
            if query.priority:
                conditions.append("priority = ?")
                params.append(query.priority.value)
            
            if query.assigned_to:
                conditions.append("assigned_to = ?")
                params.append(query.assigned_to)
            
            if query.affected_user_id:
                conditions.append("affected_user_id = ?")
                params.append(query.affected_user_id)
            
            if query.created_after:
                conditions.append("created_at >= ?")
                params.append(query.created_after.isoformat())
            
            if query.created_before:
                conditions.append("created_at <= ?")
                params.append(query.created_before.isoformat())
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            sql = f"""
                SELECT * FROM cases {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([query.limit, query.offset])
            
            cases = []
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        cases.append(self._row_to_case(row))
            
            return cases
            
        except Exception as e:
            logger.error(f"Failed to query cases: {e}")
            return []
    
    async def add_case_action(self, action: CaseAction) -> CaseAction:
        """Add an action/activity to a case"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO case_actions (
                        action_id, case_id, action_type, performed_by,
                        timestamp, description, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    action.action_id, action.case_id, action.action_type.value,
                    action.performed_by, action.timestamp.isoformat(),
                    action.description, json.dumps(action.details)
                ))
                await db.commit()
            
            return action
            
        except Exception as e:
            logger.error(f"Failed to add case action: {e}")
            raise HTTPException(status_code=500, detail="Failed to add case action")
    
    async def get_case_actions(self, case_id: str) -> List[CaseAction]:
        """Get all actions for a case"""
        try:
            actions = []
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM case_actions WHERE case_id = ? ORDER BY timestamp DESC
                """, (case_id,)) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        actions.append(CaseAction(
                            action_id=row[0],
                            case_id=row[1],
                            action_type=ActionType(row[2]),
                            performed_by=row[3],
                            timestamp=datetime.fromisoformat(row[4]),
                            description=row[5],
                            details=json.loads(row[6]) if row[6] else {}
                        ))
            
            return actions
            
        except Exception as e:
            logger.error(f"Failed to get case actions for {case_id}: {e}")
            return []
    
    async def add_evidence(self, evidence: Evidence) -> Evidence:
        """Add evidence to a case"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO evidence (
                        evidence_id, case_id, type, title, description,
                        file_path, metadata, collected_by, collected_at, hash_value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    evidence.evidence_id, evidence.case_id, evidence.type,
                    evidence.title, evidence.description, evidence.file_path,
                    json.dumps(evidence.metadata), evidence.collected_by,
                    evidence.collected_at.isoformat(), evidence.hash_value
                ))
                await db.commit()
            
            # Log evidence addition
            await self.add_case_action(CaseAction(
                case_id=evidence.case_id,
                action_type=ActionType.EVIDENCE_ADDED,
                performed_by=evidence.collected_by,
                description=f"Evidence added: {evidence.title}",
                details={"evidence_type": evidence.type, "evidence_id": evidence.evidence_id}
            ))
            
            return evidence
            
        except Exception as e:
            logger.error(f"Failed to add evidence: {e}")
            raise HTTPException(status_code=500, detail="Failed to add evidence")
    
    async def get_case_evidence(self, case_id: str) -> List[Evidence]:
        """Get all evidence for a case"""
        try:
            evidence_list = []
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM evidence WHERE case_id = ? ORDER BY collected_at DESC
                """, (case_id,)) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        evidence_list.append(Evidence(
                            evidence_id=row[0],
                            case_id=row[1],
                            type=row[2],
                            title=row[3],
                            description=row[4],
                            file_path=row[5],
                            metadata=json.loads(row[6]) if row[6] else {},
                            collected_by=row[7],
                            collected_at=datetime.fromisoformat(row[8]),
                            hash_value=row[9]
                        ))
            
            return evidence_list
            
        except Exception as e:
            logger.error(f"Failed to get evidence for case {case_id}: {e}")
            return []
    
    async def get_statistics(self) -> CaseStatistics:
        """Get case management statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Basic counts
                total_cases = await self._get_count(db, "SELECT COUNT(*) FROM cases")
                open_cases = await self._get_count(db, "SELECT COUNT(*) FROM cases WHERE status = 'open'")
                in_progress = await self._get_count(db, "SELECT COUNT(*) FROM cases WHERE status = 'in_progress'")
                resolved_cases = await self._get_count(db, "SELECT COUNT(*) FROM cases WHERE status LIKE 'closed_%'")
                
                # Overdue cases
                now = datetime.now(timezone.utc).isoformat()
                overdue_cases = await self._get_count(db, """
                    SELECT COUNT(*) FROM cases 
                    WHERE due_date < ? AND status NOT LIKE 'closed_%'
                """, (now,))
                
                # High priority cases
                high_priority = await self._get_count(db, """
                    SELECT COUNT(*) FROM cases WHERE priority IN ('high', 'critical')
                """)
                
                # Average resolution time
                avg_resolution_time = 0.0
                async with db.execute("""
                    SELECT AVG((julianday(resolved_at) - julianday(created_at)) * 24)
                    FROM cases WHERE resolved_at IS NOT NULL
                """) as cursor:
                    result = await cursor.fetchone()
                    avg_resolution_time = result[0] if result[0] else 0.0
                
                # Cases by type
                cases_by_type = {}
                async with db.execute("SELECT case_type, COUNT(*) FROM cases GROUP BY case_type") as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        cases_by_type[row[0]] = row[1]
                
                # Cases by analyst
                cases_by_analyst = {}
                async with db.execute("""
                    SELECT assigned_to, COUNT(*) FROM cases 
                    WHERE assigned_to IS NOT NULL GROUP BY assigned_to
                """) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        cases_by_analyst[row[0]] = row[1]
                
                # Monthly trend (last 12 months)
                monthly_trend = []
                for i in range(12):
                    month_start = (datetime.now(timezone.utc) - timedelta(days=30*i)).replace(day=1)
                    month_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                    
                    count = await self._get_count(db, """
                        SELECT COUNT(*) FROM cases 
                        WHERE created_at >= ? AND created_at <= ?
                    """, (month_start.isoformat(), month_end.isoformat()))
                    
                    monthly_trend.append({
                        "month": month_start.strftime("%Y-%m"),
                        "cases": count
                    })
                
                # SLA compliance
                sla_compliant = await self._get_count(db, """
                    SELECT COUNT(*) FROM cases 
                    WHERE resolved_at IS NOT NULL 
                    AND resolved_at <= due_date
                """)
                
                sla_compliance_rate = sla_compliant / resolved_cases if resolved_cases > 0 else 0.0
                
                return CaseStatistics(
                    total_cases=total_cases,
                    open_cases=open_cases,
                    in_progress_cases=in_progress,
                    resolved_cases=resolved_cases,
                    overdue_cases=overdue_cases,
                    high_priority_cases=high_priority,
                    avg_resolution_time_hours=avg_resolution_time,
                    cases_by_type=cases_by_type,
                    cases_by_analyst=cases_by_analyst,
                    monthly_trend=list(reversed(monthly_trend)),
                    sla_compliance_rate=sla_compliance_rate
                )
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return CaseStatistics(
                total_cases=0, open_cases=0, in_progress_cases=0,
                resolved_cases=0, overdue_cases=0, high_priority_cases=0,
                avg_resolution_time_hours=0.0, cases_by_type={},
                cases_by_analyst={}, monthly_trend=[], sla_compliance_rate=0.0
            )
    
    async def _generate_case_number(self) -> str:
        """Generate unique case number"""
        # In production, this would be more sophisticated
        year = datetime.now(timezone.utc).year
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT COUNT(*) FROM cases WHERE case_number LIKE ?
            """, (f"CASE-{year}-%",)) as cursor:
                count = (await cursor.fetchone())[0]
        
        return f"CASE-{year}-{count + 1:06d}"
    
    async def _get_count(self, db, query: str, params: tuple = ()) -> int:
        """Helper to get count from database"""
        async with db.execute(query, params) as cursor:
            result = await cursor.fetchone()
            return result[0] if result else 0
    
    def _row_to_case(self, row) -> FraudCase:
        """Convert database row to FraudCase object"""
        return FraudCase(
            case_id=row[0],
            case_number=row[1],
            title=row[2],
            description=row[3],
            case_type=CaseType(row[4]),
            status=CaseStatus(row[5]),
            priority=CasePriority(row[6]),
            affected_user_id=row[7],
            reported_by=row[8],
            assigned_to=row[9],
            created_at=datetime.fromisoformat(row[10]),
            updated_at=datetime.fromisoformat(row[11]),
            due_date=datetime.fromisoformat(row[12]) if row[12] else None,
            resolved_at=datetime.fromisoformat(row[13]) if row[13] else None,
            estimated_loss=row[14],
            recovered_amount=row[15],
            tags=json.loads(row[16]) if row[16] else [],
            related_cases=json.loads(row[17]) if row[17] else [],
            sla_hours=row[18] or 72,
            escalation_level=row[19] or 0,
            risk_score=row[20] or 0.0,
            confidence_level=row[21] or 0.0,
            resolution_summary=row[22],
            lessons_learned=row[23],
            preventive_measures=json.loads(row[24]) if row[24] else []
        )

# FastAPI Application
app = FastAPI(
    title="FraudGuard 360 Case Management",
    description="Fraud investigation case management system",
    version="1.0.0"
)

# Global case manager
case_manager = CaseManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await case_manager.initialize_database()
    logger.info("Case Management Service started")

@app.post("/api/v1/cases", response_model=FraudCase)
async def create_case(case: FraudCase):
    """Create a new fraud case"""
    return await case_manager.create_case(case)

@app.get("/api/v1/cases/{case_id}", response_model=FraudCase)
async def get_case(case_id: str):
    """Get a specific case"""
    case = await case_manager.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case

@app.put("/api/v1/cases/{case_id}", response_model=FraudCase)
async def update_case(case_id: str, update: CaseUpdate, updated_by: str = "system"):
    """Update an existing case"""
    case = await case_manager.update_case(case_id, update, updated_by)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case

@app.post("/api/v1/cases/query", response_model=List[FraudCase])
async def query_cases(query: CaseQuery):
    """Query cases with filters"""
    return await case_manager.query_cases(query)

@app.get("/api/v1/cases/{case_id}/actions", response_model=List[CaseAction])
async def get_case_actions(case_id: str):
    """Get all actions for a case"""
    return await case_manager.get_case_actions(case_id)

@app.post("/api/v1/cases/{case_id}/actions", response_model=CaseAction)
async def add_case_action(case_id: str, action: CaseAction):
    """Add an action to a case"""
    action.case_id = case_id
    return await case_manager.add_case_action(action)

@app.post("/api/v1/cases/{case_id}/evidence", response_model=Evidence)
async def add_evidence(case_id: str, evidence: Evidence):
    """Add evidence to a case"""
    evidence.case_id = case_id
    return await case_manager.add_evidence(evidence)

@app.get("/api/v1/cases/{case_id}/evidence", response_model=List[Evidence])
async def get_case_evidence(case_id: str):
    """Get all evidence for a case"""
    return await case_manager.get_case_evidence(case_id)

@app.get("/api/v1/cases/stats", response_model=CaseStatistics)
async def get_case_statistics():
    """Get case management statistics"""
    return await case_manager.get_statistics()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)