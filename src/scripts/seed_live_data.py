#!/usr/bin/env python3
"""
FraudGuard-360 Telecom CDR Data Seeder
======================================

Generates 1000 realistic Call Detail Records (CDRs) and seeds them into PostgreSQL.
Includes simulation of telecom fraud patterns:
- Wangiri: Short duration (1s), International callee (+220, +236), Many calls from same caller
- SIM Box: High volume outbound, same IMEI, different MSISDNs, static Cell Tower

Usage:
    python seed_live_data.py [--count 1000] [--clear]

Author: FraudGuard-360 Platform Team
License: MIT
"""

import argparse
import random
import string
import sys
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Tuple
import uuid

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.orm import Session
from shared.database import (
    get_engine, get_session_factory, init_db, Base,
    Subscriber, CellTower, CDR, Alert, GraphNode, GraphEdge,
    CDRStatus, RiskLevel, AlertSeverity, CallType, FraudType
)


# ============================================================================
# TELECOM DATA GENERATORS
# ============================================================================

# Moroccan MSISDN prefixes (local numbers)
MOROCCO_PREFIXES = ["+212", "+212", "+212"]  # Maroc Telecom, Orange, Inwi
LOCAL_PREFIXES = ["+212600", "+212661", "+212662", "+212663", "+212700", "+212701", "+212702"]

# International prefixes for Wangiri fraud (high-cost destinations)
WANGIRI_DESTINATIONS = [
    "+220",   # Gambia
    "+236",   # Central African Republic
    "+252",   # Somalia
    "+255",   # Tanzania
    "+260",   # Zambia
    "+267",   # Botswana
    "+370",   # Lithuania (common Wangiri source)
    "+371",   # Latvia
    "+372",   # Estonia
]

# Normal international destinations
NORMAL_INTERNATIONAL = [
    "+33",    # France
    "+34",    # Spain
    "+49",    # Germany
    "+1",     # USA
    "+44",    # UK
    "+971",   # UAE
]

# Cell tower locations in Morocco
CELL_TOWERS = [
    ("TWR-CAS-001", "Casablanca Centre", 33.5731, -7.5898),
    ("TWR-CAS-002", "Casablanca Port", 33.5950, -7.6187),
    ("TWR-CAS-003", "Casablanca Ain Sebaa", 33.6100, -7.5400),
    ("TWR-RAB-001", "Rabat Centre", 34.0209, -6.8416),
    ("TWR-RAB-002", "Rabat Agdal", 33.9911, -6.8498),
    ("TWR-FES-001", "Fès Medina", 34.0181, -5.0078),
    ("TWR-MRK-001", "Marrakech Gueliz", 31.6295, -8.0083),
    ("TWR-MRK-002", "Marrakech Medina", 31.6340, -7.9956),
    ("TWR-TNG-001", "Tanger Centre", 35.7595, -5.8340),
    ("TWR-AGD-001", "Agadir Centre", 30.4278, -9.5981),
    # SIM Box suspected towers (border/industrial areas)
    ("TWR-BRD-001", "Zone Industrielle Nord", 35.1200, -5.5000),
    ("TWR-BRD-002", "Zone Industrielle Sud", 30.0500, -9.2000),
]

# Subscriber names (for display)
FIRST_NAMES = [
    "Mohammed", "Fatima", "Ahmed", "Aisha", "Youssef", "Khadija", "Omar", "Zineb",
    "Hassan", "Meryem", "Rachid", "Salma", "Karim", "Nadia", "Khalid", "Hiba",
    "Samir", "Laila", "Reda", "Samira", "Amine", "Hajar", "Mehdi", "Sara"
]

LAST_NAMES = [
    "El Amrani", "Benani", "Chaoui", "Tazi", "El Idrissi", "Berrada", "Alaoui",
    "Bennani", "Fassi", "El Ouafi", "Ziane", "Belmahi", "El Mansouri", "Khaldi"
]


def generate_msisdn(prefix: str = None) -> str:
    """Generate a realistic MSISDN."""
    if prefix is None:
        prefix = random.choice(LOCAL_PREFIXES)
    suffix = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return f"{prefix}{suffix}"


def generate_imei() -> str:
    """Generate a realistic IMEI."""
    # TAC (8 digits) + Serial (6 digits) + Check digit
    tac = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    serial = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return f"{tac}{serial}0"


def generate_cdr_id() -> str:
    """Generate unique CDR ID."""
    return f"CDR-{uuid.uuid4().hex[:12].upper()}"


def generate_subscriber_name() -> str:
    """Generate realistic subscriber name."""
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# ============================================================================
# FRAUD PATTERN GENERATORS
# ============================================================================

class WangiriFraudGenerator:
    """Generate Wangiri fraud patterns."""
    
    def __init__(self, num_callers: int = 5):
        self.callers = [generate_msisdn() for _ in range(num_callers)]
        self.imeis = [generate_imei() for _ in range(num_callers)]
    
    def generate_cdr(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a Wangiri fraud CDR."""
        caller_idx = random.randint(0, len(self.callers) - 1)
        return {
            "caller_msisdn": self.callers[caller_idx],
            "callee_msisdn": generate_msisdn(random.choice(WANGIRI_DESTINATIONS)),
            "duration": random.randint(1, 3),  # Very short duration (1-3 seconds)
            "call_type": CallType.VOICE,
            "cell_tower_id": random.choice(CELL_TOWERS[:8])[0],  # Normal towers
            "roaming": False,
            "imei": self.imeis[caller_idx],
            "fraud_type": FraudType.WANGIRI,
            "timestamp": timestamp,
        }


class SIMBoxFraudGenerator:
    """Generate SIM Box fraud patterns."""
    
    def __init__(self, num_boxes: int = 3):
        # Each SIM box has one IMEI but many SIM cards (MSISDNs)
        self.boxes = []
        for _ in range(num_boxes):
            self.boxes.append({
                "imei": generate_imei(),
                "msisdns": [generate_msisdn() for _ in range(random.randint(8, 15))],
                "cell_tower": random.choice(CELL_TOWERS[-2:])[0],  # Border/industrial towers
            })
    
    def generate_cdr(self, timestamp: datetime) -> Dict[str, Any]:
        """Generate a SIM Box fraud CDR."""
        box = random.choice(self.boxes)
        return {
            "caller_msisdn": random.choice(box["msisdns"]),
            "callee_msisdn": generate_msisdn(random.choice(LOCAL_PREFIXES + NORMAL_INTERNATIONAL[:3])),
            "duration": random.randint(30, 300),  # Normal call durations
            "call_type": CallType.VOICE,
            "cell_tower_id": box["cell_tower"],  # Static tower (suspicious)
            "roaming": False,
            "imei": box["imei"],  # Same IMEI for all SIMs in box
            "fraud_type": FraudType.SIMBOX,
            "timestamp": timestamp,
        }


def calculate_risk_score(cdr_data: Dict[str, Any]) -> Tuple[float, RiskLevel, CDRStatus, str, List[Dict]]:
    """Calculate risk score and factors for a CDR."""
    score = 0.0
    factors = []
    
    fraud_type = cdr_data.get("fraud_type", FraudType.NONE)
    duration = cdr_data.get("duration", 0)
    callee = cdr_data.get("callee_msisdn", "")
    cell_tower = cdr_data.get("cell_tower_id", "")
    
    # Check for Wangiri indicators
    if fraud_type == FraudType.WANGIRI:
        score += 0.5
        factors.append({"rule": "wangiri_pattern", "value": "detected", "impact": 0.5})
    elif duration <= 3 and any(callee.startswith(dest) for dest in WANGIRI_DESTINATIONS):
        score += 0.4
        factors.append({"rule": "short_international_call", "value": f"{duration}s", "impact": 0.4})
    
    # Check for SIM Box indicators  
    if fraud_type == FraudType.SIMBOX:
        score += 0.6
        factors.append({"rule": "simbox_pattern", "value": "detected", "impact": 0.6})
    elif cell_tower in ["TWR-BRD-001", "TWR-BRD-002"]:
        score += 0.2
        factors.append({"rule": "suspicious_tower", "value": cell_tower, "impact": 0.2})
    
    # High-cost destination check
    if any(callee.startswith(dest) for dest in WANGIRI_DESTINATIONS):
        score += 0.15
        factors.append({"rule": "high_cost_destination", "value": callee[:4], "impact": 0.15})
    
    # Roaming check
    if cdr_data.get("roaming", False):
        score += 0.1
        factors.append({"rule": "roaming_call", "value": True, "impact": 0.1})
    
    score = min(score, 1.0)
    
    if score >= 0.7:
        risk_level = RiskLevel.HIGH
        status = CDRStatus.FLAGGED
        decision = "block"
    elif score >= 0.5:
        risk_level = RiskLevel.MEDIUM
        status = CDRStatus.UNDER_REVIEW
        decision = "review"
    elif score >= 0.3:
        risk_level = RiskLevel.MEDIUM
        status = CDRStatus.PENDING
        decision = "review"
    else:
        risk_level = RiskLevel.LOW
        status = CDRStatus.APPROVED
        decision = "approve"
    
    return score, risk_level, status, decision, factors


# ============================================================================
# SEEDERS
# ============================================================================

def seed_subscribers(session: Session, count: int = 100) -> List[Subscriber]:
    """Create subscribers (phone numbers)."""
    print(f"Creating {count} subscribers...")
    subscribers = []
    
    for i in range(count):
        msisdn = generate_msisdn()
        
        subscriber = Subscriber(
            msisdn=msisdn,
            imsi=f"604{random.randint(10, 99)}{random.randint(1000000000, 9999999999)}",
            name=generate_subscriber_name(),
            subscriber_type=random.choice(["prepaid", "prepaid", "prepaid", "postpaid"]),
            risk_score=0.0,
            total_calls=0,
            flagged_calls=0,
            country_code="+212",
        )
        subscribers.append(subscriber)
    
    session.add_all(subscribers)
    session.commit()
    print(f"✓ Created {len(subscribers)} subscribers")
    return subscribers


def seed_cell_towers(session: Session) -> List[CellTower]:
    """Create cell towers."""
    print(f"Creating {len(CELL_TOWERS)} cell towers...")
    towers = []
    
    for tower_id, name, lat, lng in CELL_TOWERS:
        tower = CellTower(
            tower_id=tower_id,
            name=name,
            location=name,
            latitude=lat,
            longitude=lng,
            risk_score=0.3 if "BRD" in tower_id else round(random.uniform(0, 0.1), 2),
        )
        towers.append(tower)
    
    session.add_all(towers)
    session.commit()
    print(f"✓ Created {len(towers)} cell towers")
    return towers


def seed_cdrs(
    session: Session, 
    subscribers: List[Subscriber], 
    towers: List[CellTower], 
    count: int = 1000
) -> List[CDR]:
    """Create CDRs with realistic telecom fraud patterns."""
    print(f"Creating {count} CDRs...")
    cdrs = []
    
    # Initialize fraud generators
    wangiri_gen = WangiriFraudGenerator(num_callers=5)
    simbox_gen = SIMBoxFraudGenerator(num_boxes=3)
    
    # Time range: last 30 days
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)
    
    # Distribution: 70% normal, 15% Wangiri, 15% SIM Box
    num_normal = int(count * 0.70)
    num_wangiri = int(count * 0.15)
    num_simbox = count - num_normal - num_wangiri
    
    print(f"  Generating {num_normal} normal CDRs, {num_wangiri} Wangiri, {num_simbox} SIM Box...")
    
    # Generate normal CDRs
    for i in range(num_normal):
        subscriber = random.choice(subscribers)
        tower = random.choice(towers[:10])  # Normal towers only
        
        random_seconds = random.randint(0, int((end_time - start_time).total_seconds()))
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        # Normal call patterns
        is_sms = random.random() < 0.3
        duration = 0 if is_sms else random.randint(5, 600)
        
        # Mostly local, sometimes international
        if random.random() < 0.85:
            callee = generate_msisdn()
        else:
            callee = generate_msisdn(random.choice(NORMAL_INTERNATIONAL))
        
        cdr_data = {
            "caller_msisdn": subscriber.msisdn,
            "callee_msisdn": callee,
            "duration": duration,
            "call_type": CallType.SMS if is_sms else CallType.VOICE,
            "cell_tower_id": tower.tower_id,
            "roaming": random.random() < 0.05,
            "imei": generate_imei(),
            "fraud_type": FraudType.NONE,
            "timestamp": timestamp,
        }
        
        score, risk_level, status, decision, factors = calculate_risk_score(cdr_data)
        
        cdr = CDR(
            cdr_id=generate_cdr_id(),
            caller_msisdn=subscriber.msisdn,
            callee_msisdn=callee,
            duration=duration,
            call_type=cdr_data["call_type"],
            cell_tower_id=tower.tower_id,
            roaming=cdr_data["roaming"],
            imei=cdr_data["imei"],
            fraud_type=FraudType.NONE,
            risk_score=round(score, 4),
            risk_level=risk_level,
            risk_factors=factors,
            status=status,
            decision=decision,
            timestamp=timestamp,
            created_at=timestamp,
            processed_at=timestamp + timedelta(milliseconds=random.randint(50, 500)),
            model_version="1.0.0",
            processing_time_ms=round(random.uniform(50, 500), 2),
        )
        cdrs.append(cdr)
    
    # Generate Wangiri CDRs
    for i in range(num_wangiri):
        random_seconds = random.randint(0, int((end_time - start_time).total_seconds()))
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        cdr_data = wangiri_gen.generate_cdr(timestamp)
        score, risk_level, status, decision, factors = calculate_risk_score(cdr_data)
        
        cdr = CDR(
            cdr_id=generate_cdr_id(),
            caller_msisdn=cdr_data["caller_msisdn"],
            callee_msisdn=cdr_data["callee_msisdn"],
            duration=cdr_data["duration"],
            call_type=cdr_data["call_type"],
            cell_tower_id=cdr_data["cell_tower_id"],
            roaming=cdr_data["roaming"],
            imei=cdr_data["imei"],
            fraud_type=FraudType.WANGIRI,
            risk_score=round(score, 4),
            risk_level=risk_level,
            risk_factors=factors,
            status=status,
            decision=decision,
            timestamp=timestamp,
            created_at=timestamp,
            processed_at=timestamp + timedelta(milliseconds=random.randint(50, 500)),
            model_version="1.0.0",
            processing_time_ms=round(random.uniform(50, 500), 2),
        )
        cdrs.append(cdr)
    
    # Generate SIM Box CDRs
    for i in range(num_simbox):
        random_seconds = random.randint(0, int((end_time - start_time).total_seconds()))
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        cdr_data = simbox_gen.generate_cdr(timestamp)
        score, risk_level, status, decision, factors = calculate_risk_score(cdr_data)
        
        cdr = CDR(
            cdr_id=generate_cdr_id(),
            caller_msisdn=cdr_data["caller_msisdn"],
            callee_msisdn=cdr_data["callee_msisdn"],
            duration=cdr_data["duration"],
            call_type=cdr_data["call_type"],
            cell_tower_id=cdr_data["cell_tower_id"],
            roaming=cdr_data["roaming"],
            imei=cdr_data["imei"],
            fraud_type=FraudType.SIMBOX,
            risk_score=round(score, 4),
            risk_level=risk_level,
            risk_factors=factors,
            status=status,
            decision=decision,
            timestamp=timestamp,
            created_at=timestamp,
            processed_at=timestamp + timedelta(milliseconds=random.randint(50, 500)),
            model_version="1.0.0",
            processing_time_ms=round(random.uniform(50, 500), 2),
        )
        cdrs.append(cdr)
    
    session.add_all(cdrs)
    session.commit()
    
    # Update subscriber stats
    for subscriber in subscribers:
        subscriber_cdrs = [c for c in cdrs if c.caller_msisdn == subscriber.msisdn]
        subscriber.total_calls = len(subscriber_cdrs)
        subscriber.flagged_calls = len([c for c in subscriber_cdrs if c.status in [CDRStatus.FLAGGED, CDRStatus.UNDER_REVIEW]])
        if subscriber.total_calls > 0:
            subscriber.risk_score = round(subscriber.flagged_calls / subscriber.total_calls, 2)
    session.commit()
    
    print(f"✓ Created {len(cdrs)} CDRs")
    return cdrs


def seed_alerts(session: Session, cdrs: List[CDR]) -> List[Alert]:
    """Create alerts for flagged CDRs."""
    print("Creating alerts for flagged CDRs...")
    alerts = []
    
    alert_templates = {
        FraudType.WANGIRI: [
            ("Wangiri Fraud Detected", "Short duration call to high-cost international destination", AlertSeverity.CRITICAL, "wangiri_detected"),
            ("Premium Rate Callback Attempt", "Possible Wangiri callback scheme detected", AlertSeverity.ERROR, "wangiri_callback"),
        ],
        FraudType.SIMBOX: [
            ("SIM Box Fraud Detected", "Multiple MSISDNs detected from same IMEI", AlertSeverity.CRITICAL, "simbox_detected"),
            ("Bypass Fraud Alert", "International call bypass through unauthorized gateway", AlertSeverity.ERROR, "simbox_bypass"),
        ],
        FraudType.NONE: [
            ("Suspicious Call Pattern", "Unusual calling behavior detected", AlertSeverity.WARNING, "pattern_anomaly"),
            ("High-Cost Destination", "Call to premium rate number detected", AlertSeverity.INFO, "high_cost_call"),
        ],
    }
    
    flagged_cdrs = [c for c in cdrs if c.status in [CDRStatus.FLAGGED, CDRStatus.UNDER_REVIEW]]
    
    for cdr in flagged_cdrs:
        templates = alert_templates.get(cdr.fraud_type, alert_templates[FraudType.NONE])
        template = random.choice(templates)
        
        alert = Alert(
            alert_id=f"ALR-{uuid.uuid4().hex[:12].upper()}",
            cdr_id=cdr.cdr_id,
            title=template[0],
            description=f"{template[1]}. CDR ID: {cdr.cdr_id}, Caller: {cdr.caller_msisdn}, Duration: {cdr.duration}s",
            severity=template[2],
            category=template[3],
            is_read=random.random() < 0.3,
            is_resolved=random.random() < 0.2,
            created_at=cdr.timestamp + timedelta(seconds=random.randint(1, 60)),
        )
        alerts.append(alert)
    
    session.add_all(alerts)
    session.commit()
    print(f"✓ Created {len(alerts)} alerts")
    return alerts


def seed_graph_data(
    session: Session, 
    subscribers: List[Subscriber], 
    towers: List[CellTower], 
    cdrs: List[CDR]
) -> None:
    """Create graph nodes and edges for visualization."""
    print("Creating graph nodes and edges...")
    
    nodes = []
    edges = []
    
    # Create MSISDN nodes from subscribers
    for subscriber in subscribers:
        node = GraphNode(
            node_id=subscriber.msisdn,
            node_type="msisdn",
            label=subscriber.msisdn,
            properties={"name": subscriber.name, "type": subscriber.subscriber_type},
            risk_score=subscriber.risk_score,
        )
        nodes.append(node)
    
    # Create Cell Tower nodes
    for tower in towers:
        node = GraphNode(
            node_id=tower.tower_id,
            node_type="cell_tower",
            label=tower.name,
            properties={"location": tower.location, "lat": tower.latitude, "lng": tower.longitude},
            risk_score=tower.risk_score,
        )
        nodes.append(node)
    
    # Track unique IMEI and callee nodes
    imeis_seen = set()
    callees_seen = set()
    
    # Create edges from CDRs
    for cdr in cdrs:
        # Caller -> Callee edge (call relationship)
        if cdr.callee_msisdn not in callees_seen:
            callees_seen.add(cdr.callee_msisdn)
            callee_node = GraphNode(
                node_id=cdr.callee_msisdn,
                node_type="msisdn",
                label=cdr.callee_msisdn,
                properties={"type": "external" if not cdr.callee_msisdn.startswith("+212") else "local"},
                risk_score=0.5 if any(cdr.callee_msisdn.startswith(d) for d in WANGIRI_DESTINATIONS) else 0.0,
            )
            nodes.append(callee_node)
        
        # Call edge
        call_edge = GraphEdge(
            source_id=cdr.caller_msisdn,
            target_id=cdr.callee_msisdn,
            edge_type="called",
            weight=cdr.duration / 60 if cdr.duration > 0 else 0.1,
            properties={"cdr_id": cdr.cdr_id, "duration": cdr.duration, "fraud_type": cdr.fraud_type.value},
        )
        edges.append(call_edge)
        
        # Create IMEI node if new
        if cdr.imei and cdr.imei not in imeis_seen:
            imeis_seen.add(cdr.imei)
            imei_node = GraphNode(
                node_id=cdr.imei,
                node_type="imei",
                label=f"IMEI:{cdr.imei[-4:]}",
                risk_score=0.0,
            )
            nodes.append(imei_node)
        
        # Caller -> IMEI edge
        if cdr.imei:
            imei_edge = GraphEdge(
                source_id=cdr.caller_msisdn,
                target_id=cdr.imei,
                edge_type="used_imei",
                weight=1.0,
            )
            edges.append(imei_edge)
        
        # Caller -> Cell Tower edge
        if cdr.cell_tower_id:
            tower_edge = GraphEdge(
                source_id=cdr.caller_msisdn,
                target_id=cdr.cell_tower_id,
                edge_type="used_tower",
                weight=1.0,
            )
            edges.append(tower_edge)
    
    session.add_all(nodes)
    session.add_all(edges)
    session.commit()
    print(f"✓ Created {len(nodes)} graph nodes and {len(edges)} edges")


def clear_all_data(session: Session) -> None:
    """Clear all existing data."""
    print("Clearing existing data...")
    session.query(GraphEdge).delete()
    session.query(GraphNode).delete()
    session.query(Alert).delete()
    session.query(CDR).delete()
    session.query(CellTower).delete()
    session.query(Subscriber).delete()
    session.commit()
    print("✓ All data cleared")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Seed FraudGuard-360 database with telecom CDR data")
    parser.add_argument("--count", type=int, default=1000, help="Number of CDRs to create")
    parser.add_argument("--subscribers", type=int, default=100, help="Number of subscribers to create")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before seeding")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FraudGuard-360 Telecom CDR Data Seeder")
    print("=" * 60)
    
    # Initialize database
    print("\nInitializing database...")
    try:
        engine = init_db()
        print("✓ Database tables created")
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        print("Make sure PostgreSQL is running and connection settings are correct.")
        sys.exit(1)
    
    # Create session
    SessionLocal = get_session_factory()
    session = SessionLocal()
    
    try:
        if args.clear:
            clear_all_data(session)
        
        print(f"\nSeeding data with {args.count} CDRs...")
        print("-" * 40)
        
        # Seed in order
        subscribers = seed_subscribers(session, args.subscribers)
        towers = seed_cell_towers(session)
        cdrs = seed_cdrs(session, subscribers, towers, args.count)
        alerts = seed_alerts(session, cdrs)
        seed_graph_data(session, subscribers, towers, cdrs)
        
        # Print summary
        print("\n" + "=" * 60)
        print("SEEDING COMPLETE!")
        print("=" * 60)
        print(f"  Subscribers:  {len(subscribers)}")
        print(f"  Cell Towers:  {len(towers)}")
        print(f"  CDRs:         {len(cdrs)}")
        print(f"  Alerts:       {len(alerts)}")
        
        # Stats
        wangiri = len([c for c in cdrs if c.fraud_type == FraudType.WANGIRI])
        simbox = len([c for c in cdrs if c.fraud_type == FraudType.SIMBOX])
        normal = len([c for c in cdrs if c.fraud_type == FraudType.NONE])
        
        flagged = len([c for c in cdrs if c.status == CDRStatus.FLAGGED])
        review = len([c for c in cdrs if c.status == CDRStatus.UNDER_REVIEW])
        approved = len([c for c in cdrs if c.status == CDRStatus.APPROVED])
        
        print(f"\nFraud Type Distribution:")
        print(f"  Normal:     {normal} ({normal/len(cdrs)*100:.1f}%)")
        print(f"  Wangiri:    {wangiri} ({wangiri/len(cdrs)*100:.1f}%)")
        print(f"  SIM Box:    {simbox} ({simbox/len(cdrs)*100:.1f}%)")
        
        print(f"\nCDR Status Breakdown:")
        print(f"  Approved:     {approved} ({approved/len(cdrs)*100:.1f}%)")
        print(f"  Under Review: {review} ({review/len(cdrs)*100:.1f}%)")
        print(f"  Flagged:      {flagged} ({flagged/len(cdrs)*100:.1f}%)")
        
        high_risk = len([c for c in cdrs if c.risk_level == RiskLevel.HIGH])
        medium_risk = len([c for c in cdrs if c.risk_level == RiskLevel.MEDIUM])
        low_risk = len([c for c in cdrs if c.risk_level == RiskLevel.LOW])
        
        print(f"\nRisk Distribution:")
        print(f"  Low:    {low_risk} ({low_risk/len(cdrs)*100:.1f}%)")
        print(f"  Medium: {medium_risk} ({medium_risk/len(cdrs)*100:.1f}%)")
        print(f"  High:   {high_risk} ({high_risk/len(cdrs)*100:.1f}%)")
        
        # Call stats
        voice_calls = len([c for c in cdrs if c.call_type == CallType.VOICE])
        sms_calls = len([c for c in cdrs if c.call_type == CallType.SMS])
        avg_duration = sum(c.duration for c in cdrs if c.call_type == CallType.VOICE) / max(voice_calls, 1)
        
        print(f"\nCall Statistics:")
        print(f"  Voice Calls:  {voice_calls}")
        print(f"  SMS:          {sms_calls}")
        print(f"  Avg Duration: {avg_duration:.1f}s")
        
    except Exception as e:
        print(f"\n✗ Error during seeding: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
