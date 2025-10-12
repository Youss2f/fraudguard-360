"""
FraudGuard 360° - Graph Analytics Service
Neo4j-powered fraud network analysis and pattern detection
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from neo4j import GraphDatabase
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetworkNode:
    """Graph network node representation"""
    node_id: str
    node_type: str  # USER, MERCHANT, DEVICE, IP_ADDRESS, ACCOUNT
    properties: Dict[str, Any]
    risk_score: float
    centrality_measures: Dict[str, float]
    community_id: Optional[str] = None

@dataclass
class NetworkRelationship:
    """Graph relationship representation"""
    source_id: str
    target_id: str
    relationship_type: str  # TRANSACTS_WITH, USES_DEVICE, FROM_IP, etc.
    properties: Dict[str, Any]
    weight: float
    risk_score: float

@dataclass
class FraudRing:
    """Detected fraud ring/community"""
    ring_id: str
    member_nodes: List[str]
    confidence_score: float
    risk_level: str
    description: str
    key_indicators: List[str]
    estimated_loss: float
    detection_timestamp: datetime

@dataclass
class GraphAnalysisResult:
    """Result of graph analysis"""
    analysis_id: str
    target_node_id: str
    fraud_probability: float
    network_risk_score: float
    suspicious_connections: List[Dict[str, Any]]
    community_analysis: Dict[str, Any]
    path_analysis: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime

class Neo4jGraphAnalyzer:
    """
    Neo4j-based graph analytics for fraud detection
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
        self.logger = logging.getLogger(__name__)
        
    async def initialize_graph_schema(self):
        """Initialize Neo4j schema and constraints"""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
                "CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.merchant_id IS UNIQUE",
                "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE",
                "CREATE CONSTRAINT ip_id IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.ip_address IS UNIQUE",
                "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
                "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    self.logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    self.logger.warning(f"Constraint already exists or failed: {e}")
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX user_risk_score IF NOT EXISTS FOR (u:User) ON (u.risk_score)",
                "CREATE INDEX merchant_risk_score IF NOT EXISTS FOR (m:Merchant) ON (m.risk_score)",
                "CREATE INDEX transaction_amount IF NOT EXISTS FOR (t:Transaction) ON (t.amount)",
                "CREATE INDEX transaction_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    self.logger.info(f"Created index: {index}")
                except Exception as e:
                    self.logger.warning(f"Index already exists or failed: {e}")
    
    async def add_transaction_to_graph(self, transaction_data: Dict[str, Any]) -> bool:
        """Add a transaction and related entities to the graph"""
        try:
            with self.driver.session() as session:
                # Cypher query to create/update nodes and relationships
                query = """
                // Create or update User node
                MERGE (u:User {user_id: $user_id})
                ON CREATE SET u.created_at = datetime(),
                              u.risk_score = $user_risk_score,
                              u.account_age_days = $account_age_days
                ON MATCH SET u.last_transaction = datetime(),
                             u.transaction_count = coalesce(u.transaction_count, 0) + 1
                
                // Create or update Merchant node
                MERGE (m:Merchant {merchant_id: $merchant_id})
                ON CREATE SET m.created_at = datetime(),
                              m.category = $merchant_category,
                              m.risk_score = $merchant_risk_score
                ON MATCH SET m.last_transaction = datetime(),
                             m.transaction_count = coalesce(m.transaction_count, 0) + 1
                
                // Create or update Device node
                MERGE (d:Device {device_id: $device_fingerprint})
                ON CREATE SET d.created_at = datetime(),
                              d.risk_score = $device_risk_score
                ON MATCH SET d.last_seen = datetime(),
                             d.usage_count = coalesce(d.usage_count, 0) + 1
                
                // Create or update IP Address node
                MERGE (ip:IPAddress {ip_address: $ip_address})
                ON CREATE SET ip.created_at = datetime(),
                              ip.reputation_score = $ip_reputation_score
                ON MATCH SET ip.last_seen = datetime(),
                             ip.usage_count = coalesce(ip.usage_count, 0) + 1
                
                // Create Transaction node
                CREATE (t:Transaction {
                    transaction_id: $transaction_id,
                    amount: $amount,
                    timestamp: datetime($timestamp),
                    fraud_score: $fraud_score,
                    risk_level: $risk_level
                })
                
                // Create relationships
                CREATE (u)-[:MADE_TRANSACTION {timestamp: datetime($timestamp), amount: $amount}]->(t)
                CREATE (t)-[:TO_MERCHANT {timestamp: datetime($timestamp), amount: $amount}]->(m)
                CREATE (u)-[:USES_DEVICE {timestamp: datetime($timestamp)}]->(d)
                CREATE (u)-[:FROM_IP {timestamp: datetime($timestamp)}]->(ip)
                
                RETURN t.transaction_id as transaction_id
                """
                
                parameters = {
                    'user_id': transaction_data['user_id'],
                    'merchant_id': transaction_data['merchant_id'],
                    'device_fingerprint': transaction_data.get('device_fingerprint', 'unknown'),
                    'ip_address': transaction_data.get('ip_address', 'unknown'),
                    'transaction_id': transaction_data['transaction_id'],
                    'amount': float(transaction_data['amount']),
                    'timestamp': transaction_data.get('timestamp', datetime.now().isoformat()),
                    'merchant_category': transaction_data.get('merchant_category', 'unknown'),
                    'fraud_score': transaction_data.get('fraud_score', 0.0),
                    'risk_level': transaction_data.get('risk_level', 'LOW'),
                    'user_risk_score': transaction_data.get('user_risk_score', 0.3),
                    'merchant_risk_score': transaction_data.get('merchant_risk_score', 0.2),
                    'device_risk_score': transaction_data.get('device_risk_score', 0.2),
                    'ip_reputation_score': transaction_data.get('ip_reputation_score', 0.8),
                    'account_age_days': transaction_data.get('account_age_days', 365)
                }
                
                result = session.run(query, parameters)
                record = result.single()
                
                if record:
                    self.logger.info(f"Added transaction {record['transaction_id']} to graph")
                    return True
                else:
                    self.logger.error("Failed to add transaction to graph")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error adding transaction to graph: {e}")
            return False
    
    async def analyze_user_network(self, user_id: str, max_depth: int = 3) -> GraphAnalysisResult:
        """Analyze user's network for fraud patterns"""
        try:
            with self.driver.session() as session:
                # Get user's transaction network
                network_query = """
                MATCH (u:User {user_id: $user_id})
                OPTIONAL MATCH path = (u)-[:MADE_TRANSACTION*1..2]-(t:Transaction)-[:TO_MERCHANT]-(m:Merchant)
                OPTIONAL MATCH (u)-[:USES_DEVICE]-(d:Device)
                OPTIONAL MATCH (u)-[:FROM_IP]-(ip:IPAddress)
                
                WITH u, collect(DISTINCT t) as transactions, 
                     collect(DISTINCT m) as merchants,
                     collect(DISTINCT d) as devices,
                     collect(DISTINCT ip) as ips
                
                // Get connected users through shared resources
                OPTIONAL MATCH (u)-[:USES_DEVICE]-(shared_device:Device)-[:USES_DEVICE]-(other_user:User)
                WHERE other_user.user_id <> $user_id
                
                OPTIONAL MATCH (u)-[:FROM_IP]-(shared_ip:IPAddress)-[:FROM_IP]-(other_user2:User)
                WHERE other_user2.user_id <> $user_id
                
                RETURN u.user_id as user_id,
                       u.risk_score as user_risk_score,
                       size(transactions) as transaction_count,
                       [t IN transactions | {id: t.transaction_id, amount: t.amount, fraud_score: t.fraud_score}] as transaction_details,
                       [m IN merchants | {id: m.merchant_id, risk_score: m.risk_score, category: m.category}] as merchant_details,
                       [d IN devices | {id: d.device_id, risk_score: d.risk_score}] as device_details,
                       [ip IN ips | {address: ip.ip_address, reputation: ip.reputation_score}] as ip_details,
                       collect(DISTINCT other_user.user_id) as device_connected_users,
                       collect(DISTINCT other_user2.user_id) as ip_connected_users
                """
                
                result = session.run(network_query, {'user_id': user_id})
                record = result.single()
                
                if not record:
                    raise HTTPException(status_code=404, detail=f"User {user_id} not found in graph")
                
                # Calculate network risk metrics
                network_risk_score = await self._calculate_network_risk(record)
                
                # Detect suspicious connections
                suspicious_connections = await self._detect_suspicious_connections(session, user_id)
                
                # Community analysis
                community_analysis = await self._analyze_user_community(session, user_id)
                
                # Path analysis to known fraudulent entities
                path_analysis = await self._analyze_fraud_paths(session, user_id)
                
                # Generate recommendations
                recommendations = self._generate_network_recommendations(
                    network_risk_score, suspicious_connections, community_analysis
                )
                
                return GraphAnalysisResult(
                    analysis_id=f"graph_analysis_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    target_node_id=user_id,
                    fraud_probability=min(network_risk_score / 100, 1.0),
                    network_risk_score=network_risk_score,
                    suspicious_connections=suspicious_connections,
                    community_analysis=community_analysis,
                    path_analysis=path_analysis,
                    recommendations=recommendations,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing user network for {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Network analysis failed: {str(e)}")
    
    async def _calculate_network_risk(self, network_data: Dict) -> float:
        """Calculate overall network risk score"""
        base_risk = network_data.get('user_risk_score', 0.3) * 100
        
        # Device sharing risk
        device_connections = len(network_data.get('device_connected_users', []))
        device_risk = min(device_connections * 10, 30)  # Up to 30 points
        
        # IP sharing risk
        ip_connections = len(network_data.get('ip_connected_users', []))
        ip_risk = min(ip_connections * 8, 25)  # Up to 25 points
        
        # Merchant risk
        merchants = network_data.get('merchant_details', [])
        avg_merchant_risk = np.mean([m.get('risk_score', 0.3) for m in merchants]) if merchants else 0.3
        merchant_risk = avg_merchant_risk * 20  # Up to 20 points
        
        # Transaction pattern risk
        transactions = network_data.get('transaction_details', [])
        avg_fraud_score = np.mean([t.get('fraud_score', 0.3) for t in transactions]) if transactions else 0.3
        transaction_risk = avg_fraud_score * 25  # Up to 25 points
        
        total_risk = base_risk + device_risk + ip_risk + merchant_risk + transaction_risk
        return min(total_risk, 100)
    
    async def _detect_suspicious_connections(self, session, user_id: str) -> List[Dict[str, Any]]:
        """Detect suspicious network connections"""
        query = """
        MATCH (u:User {user_id: $user_id})
        
        // Find users sharing devices with high fraud scores
        OPTIONAL MATCH (u)-[:USES_DEVICE]-(d:Device)-[:USES_DEVICE]-(suspicious_user:User)
        WHERE suspicious_user.user_id <> $user_id 
        AND suspicious_user.risk_score > 0.7
        
        // Find users sharing IPs with bad reputation
        OPTIONAL MATCH (u)-[:FROM_IP]-(ip:IPAddress)-[:FROM_IP]-(ip_suspicious_user:User)
        WHERE ip_suspicious_user.user_id <> $user_id 
        AND ip.reputation_score < 0.3
        
        // Find connections to merchants with high fraud rates
        OPTIONAL MATCH (u)-[:MADE_TRANSACTION]-(t:Transaction)-[:TO_MERCHANT]-(m:Merchant)
        WHERE m.risk_score > 0.8
        
        RETURN collect(DISTINCT {
            type: 'device_sharing',
            connected_user: suspicious_user.user_id,
            risk_score: suspicious_user.risk_score,
            shared_resource: d.device_id
        }) as device_connections,
        collect(DISTINCT {
            type: 'ip_sharing',
            connected_user: ip_suspicious_user.user_id,
            risk_score: ip_suspicious_user.risk_score,
            shared_resource: ip.ip_address,
            ip_reputation: ip.reputation_score
        }) as ip_connections,
        collect(DISTINCT {
            type: 'high_risk_merchant',
            merchant_id: m.merchant_id,
            merchant_risk: m.risk_score,
            category: m.category
        }) as merchant_connections
        """
        
        result = session.run(query, {'user_id': user_id})
        record = result.single()
        
        if not record:
            return []
        
        suspicious_connections = []
        suspicious_connections.extend(record.get('device_connections', []))
        suspicious_connections.extend(record.get('ip_connections', []))
        suspicious_connections.extend(record.get('merchant_connections', []))
        
        # Filter out null/empty connections
        return [conn for conn in suspicious_connections if conn and any(conn.values())]
    
    async def _analyze_user_community(self, session, user_id: str) -> Dict[str, Any]:
        """Analyze user's community/cluster in the network"""
        query = """
        MATCH (u:User {user_id: $user_id})
        
        // Get users within 2 hops through various relationships
        OPTIONAL MATCH (u)-[:USES_DEVICE|FROM_IP|MADE_TRANSACTION*1..2]-(connected_user:User)
        WHERE connected_user.user_id <> $user_id
        
        WITH u, collect(DISTINCT connected_user) as community_members
        
        // Calculate community statistics
        WITH u, community_members,
             [member IN community_members | member.risk_score] as risk_scores
        
        RETURN size(community_members) as community_size,
               CASE WHEN size(risk_scores) > 0 
                    THEN reduce(sum = 0.0, score IN risk_scores | sum + score) / size(risk_scores)
                    ELSE 0.0 
               END as avg_community_risk,
               CASE WHEN size(risk_scores) > 0
                    THEN size([score IN risk_scores WHERE score > 0.7])
                    ELSE 0
               END as high_risk_members
        """
        
        result = session.run(query, {'user_id': user_id})
        record = result.single()
        
        if not record:
            return {'community_size': 0, 'avg_community_risk': 0.0, 'high_risk_members': 0}
        
        return {
            'community_size': record['community_size'],
            'avg_community_risk': record['avg_community_risk'],
            'high_risk_members': record['high_risk_members'],
            'community_risk_level': 'HIGH' if record['avg_community_risk'] > 0.6 else 
                                   'MEDIUM' if record['avg_community_risk'] > 0.4 else 'LOW'
        }
    
    async def _analyze_fraud_paths(self, session, user_id: str) -> Dict[str, Any]:
        """Analyze paths to known fraudulent entities"""
        query = """
        MATCH (u:User {user_id: $user_id})
        
        // Find shortest paths to known fraudulent users
        OPTIONAL MATCH path = shortestPath((u)-[:USES_DEVICE|FROM_IP|MADE_TRANSACTION*1..4]-(fraud_user:User))
        WHERE fraud_user.user_id <> $user_id 
        AND fraud_user.risk_score > 0.9
        
        // Find paths to blacklisted merchants
        OPTIONAL MATCH merchant_path = shortestPath((u)-[:MADE_TRANSACTION*1..3]-(blacklisted_merchant:Merchant))
        WHERE blacklisted_merchant.risk_score > 0.95
        
        RETURN collect(DISTINCT {
            target: fraud_user.user_id,
            path_length: length(path),
            target_risk: fraud_user.risk_score
        }) as fraud_user_paths,
        collect(DISTINCT {
            target: blacklisted_merchant.merchant_id,
            path_length: length(merchant_path),
            target_risk: blacklisted_merchant.risk_score
        }) as blacklisted_merchant_paths
        """
        
        result = session.run(query, {'user_id': user_id})
        record = result.single()
        
        if not record:
            return {'fraud_user_paths': [], 'blacklisted_merchant_paths': []}
        
        fraud_paths = record.get('fraud_user_paths', [])
        merchant_paths = record.get('blacklisted_merchant_paths', [])
        
        # Calculate minimum path lengths
        min_fraud_path = min([p['path_length'] for p in fraud_paths], default=float('inf'))
        min_merchant_path = min([p['path_length'] for p in merchant_paths], default=float('inf'))
        
        return {
            'fraud_user_paths': fraud_paths,
            'blacklisted_merchant_paths': merchant_paths,
            'min_fraud_distance': min_fraud_path if min_fraud_path != float('inf') else None,
            'min_blacklist_distance': min_merchant_path if min_merchant_path != float('inf') else None,
            'has_close_fraud_connections': min_fraud_path <= 2 or min_merchant_path <= 2
        }
    
    def _generate_network_recommendations(self, network_risk: float, suspicious_connections: List, 
                                        community_analysis: Dict) -> List[str]:
        """Generate recommendations based on network analysis"""
        recommendations = []
        
        if network_risk > 80:
            recommendations.append("BLOCK_ALL_TRANSACTIONS")
            recommendations.append("FREEZE_ACCOUNT_IMMEDIATELY")
            recommendations.append("INVESTIGATE_FRAUD_RING")
        elif network_risk > 60:
            recommendations.append("REQUIRE_ENHANCED_VERIFICATION")
            recommendations.append("MONITOR_ALL_TRANSACTIONS")
            recommendations.append("REVIEW_ACCOUNT_CONNECTIONS")
        elif network_risk > 40:
            recommendations.append("INCREASE_TRANSACTION_LIMITS_MONITORING")
            recommendations.append("FLAG_SUSPICIOUS_PATTERNS")
        else:
            recommendations.append("STANDARD_MONITORING")
        
        # Connection-specific recommendations
        if len(suspicious_connections) > 3:
            recommendations.append("INVESTIGATE_DEVICE_SHARING")
        
        if community_analysis.get('high_risk_members', 0) > 5:
            recommendations.append("ANALYZE_FRAUD_RING_MEMBERSHIP")
        
        return recommendations
    
    async def detect_fraud_rings(self, min_ring_size: int = 3, max_rings: int = 10) -> List[FraudRing]:
        """Detect potential fraud rings using community detection"""
        try:
            with self.driver.session() as session:
                # Community detection query using shared devices and IPs
                query = """
                // Find clusters of users sharing devices or IPs
                MATCH (u1:User)-[:USES_DEVICE|FROM_IP]-(resource)-[:USES_DEVICE|FROM_IP]-(u2:User)
                WHERE u1.user_id < u2.user_id  // Avoid duplicates
                AND (u1.risk_score > 0.5 OR u2.risk_score > 0.5)  // At least one high-risk user
                
                WITH u1, u2, count(resource) as shared_resources
                WHERE shared_resources >= 2  // Minimum shared resources
                
                // Build connected components
                WITH collect({user1: u1.user_id, user2: u2.user_id, weight: shared_resources}) as connections
                
                UNWIND connections as conn
                MATCH (u1:User {user_id: conn.user1}), (u2:User {user_id: conn.user2})
                
                WITH collect(DISTINCT u1.user_id) + collect(DISTINCT u2.user_id) as all_users
                
                // Get user details for analysis
                UNWIND all_users as user_id
                MATCH (u:User {user_id: user_id})
                
                RETURN collect({
                    user_id: u.user_id,
                    risk_score: u.risk_score,
                    transaction_count: u.transaction_count
                }) as potential_ring_members
                """
                
                result = session.run(query)
                records = result.data()
                
                if not records:
                    return []
                
                fraud_rings = []
                ring_counter = 1
                
                for record in records[:max_rings]:
                    members = record.get('potential_ring_members', [])
                    
                    if len(members) >= min_ring_size:
                        # Calculate ring metrics
                        avg_risk = np.mean([m['risk_score'] for m in members])
                        total_transactions = sum([m.get('transaction_count', 0) for m in members])
                        
                        # Determine confidence and risk level
                        confidence = min(avg_risk + (len(members) / 20), 1.0)
                        risk_level = 'CRITICAL' if avg_risk > 0.8 else 'HIGH' if avg_risk > 0.6 else 'MEDIUM'
                        
                        # Key indicators
                        indicators = [
                            f"Shared resources among {len(members)} users",
                            f"Average risk score: {avg_risk:.2f}",
                            f"Total transactions: {total_transactions}"
                        ]
                        
                        fraud_ring = FraudRing(
                            ring_id=f"fraud_ring_{ring_counter:03d}",
                            member_nodes=[m['user_id'] for m in members],
                            confidence_score=confidence,
                            risk_level=risk_level,
                            description=f"Detected fraud ring with {len(members)} members sharing devices/IPs",
                            key_indicators=indicators,
                            estimated_loss=total_transactions * avg_risk * 100,  # Rough estimate
                            detection_timestamp=datetime.now()
                        )
                        
                        fraud_rings.append(fraud_ring)
                        ring_counter += 1
                
                return fraud_rings
                
        except Exception as e:
            self.logger.error(f"Error detecting fraud rings: {e}")
            return []
    
    async def get_network_statistics(self) -> Dict[str, Any]:
        """Get overall network statistics"""
        try:
            with self.driver.session() as session:
                query = """
                // Get basic network statistics
                MATCH (u:User) 
                WITH count(u) as total_users
                
                MATCH (m:Merchant)
                WITH total_users, count(m) as total_merchants
                
                MATCH (t:Transaction)
                WITH total_users, total_merchants, count(t) as total_transactions,
                     sum(t.amount) as total_volume
                
                MATCH (d:Device)
                WITH total_users, total_merchants, total_transactions, total_volume,
                     count(d) as total_devices
                
                MATCH (ip:IPAddress)
                WITH total_users, total_merchants, total_transactions, total_volume,
                     total_devices, count(ip) as total_ips
                
                // Risk statistics
                MATCH (u:User)
                WITH total_users, total_merchants, total_transactions, total_volume,
                     total_devices, total_ips,
                     avg(u.risk_score) as avg_user_risk,
                     count(CASE WHEN u.risk_score > 0.7 THEN 1 END) as high_risk_users
                
                MATCH (m:Merchant)
                WITH total_users, total_merchants, total_transactions, total_volume,
                     total_devices, total_ips, avg_user_risk, high_risk_users,
                     avg(m.risk_score) as avg_merchant_risk,
                     count(CASE WHEN m.risk_score > 0.7 THEN 1 END) as high_risk_merchants
                
                RETURN total_users, total_merchants, total_transactions, total_volume,
                       total_devices, total_ips, avg_user_risk, high_risk_users,
                       avg_merchant_risk, high_risk_merchants
                """
                
                result = session.run(query)
                record = result.single()
                
                if record:
                    return {
                        'network_size': {
                            'total_users': record['total_users'],
                            'total_merchants': record['total_merchants'],
                            'total_devices': record['total_devices'],
                            'total_ips': record['total_ips'],
                            'total_transactions': record['total_transactions']
                        },
                        'business_metrics': {
                            'total_transaction_volume': record['total_volume'],
                            'avg_transaction_value': record['total_volume'] / max(record['total_transactions'], 1)
                        },
                        'risk_metrics': {
                            'avg_user_risk_score': record['avg_user_risk'],
                            'avg_merchant_risk_score': record['avg_merchant_risk'],
                            'high_risk_users': record['high_risk_users'],
                            'high_risk_merchants': record['high_risk_merchants'],
                            'user_risk_percentage': (record['high_risk_users'] / max(record['total_users'], 1)) * 100,
                            'merchant_risk_percentage': (record['high_risk_merchants'] / max(record['total_merchants'], 1)) * 100
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {'error': 'No data found in graph database'}
                    
        except Exception as e:
            self.logger.error(f"Error getting network statistics: {e}")
            return {'error': str(e)}
    
    def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()

# FastAPI Application
app = FastAPI(
    title="FraudGuard 360° - Graph Analytics Service",
    description="Neo4j-powered fraud network analysis and pattern detection",
    version="1.0.0"
)

# Global graph analyzer
graph_analyzer: Optional[Neo4jGraphAnalyzer] = None

@app.on_event("startup")
async def startup_event():
    """Initialize graph analyzer on startup"""
    global graph_analyzer
    graph_analyzer = Neo4jGraphAnalyzer()
    await graph_analyzer.initialize_graph_schema()
    logger.info("Graph Analytics Service initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if graph_analyzer:
        graph_analyzer.close()
    logger.info("Graph Analytics Service shut down")

# Pydantic models
class TransactionGraphRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    transaction_id: str = Field(..., description="Transaction identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., ge=0, description="Transaction amount")
    device_fingerprint: Optional[str] = Field(None, description="Device fingerprint")
    ip_address: Optional[str] = Field(None, description="IP address")
    fraud_score: Optional[float] = Field(0.0, ge=0, le=1, description="Fraud score")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)

class NetworkAnalysisRequest(BaseModel):
    user_id: str = Field(..., description="User to analyze")
    analysis_depth: int = Field(3, ge=1, le=5, description="Analysis depth (1-5)")

class FraudRingDetectionRequest(BaseModel):
    min_ring_size: int = Field(3, ge=2, le=20, description="Minimum ring size")
    max_rings: int = Field(10, ge=1, le=50, description="Maximum rings to return")

# API Endpoints
@app.post("/add_transaction")
async def add_transaction_to_graph(request: TransactionGraphRequest):
    """Add a transaction to the graph database"""
    if not graph_analyzer:
        raise HTTPException(status_code=503, detail="Graph service not initialized")
    
    transaction_data = request.dict()
    success = await graph_analyzer.add_transaction_to_graph(transaction_data)
    
    if success:
        return {"message": "Transaction added to graph", "transaction_id": request.transaction_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to add transaction to graph")

@app.post("/analyze_network")
async def analyze_user_network(request: NetworkAnalysisRequest):
    """Analyze user's network for fraud patterns"""
    if not graph_analyzer:
        raise HTTPException(status_code=503, detail="Graph service not initialized")
    
    analysis_result = await graph_analyzer.analyze_user_network(
        request.user_id, request.analysis_depth
    )
    
    return analysis_result

@app.post("/detect_fraud_rings")
async def detect_fraud_rings(request: FraudRingDetectionRequest):
    """Detect potential fraud rings in the network"""
    if not graph_analyzer:
        raise HTTPException(status_code=503, detail="Graph service not initialized")
    
    fraud_rings = await graph_analyzer.detect_fraud_rings(
        request.min_ring_size, request.max_rings
    )
    
    return {
        "fraud_rings_detected": len(fraud_rings),
        "fraud_rings": fraud_rings
    }

@app.get("/network_statistics")
async def get_network_statistics():
    """Get network statistics and health metrics"""
    if not graph_analyzer:
        raise HTTPException(status_code=503, detail="Graph service not initialized")
    
    stats = await graph_analyzer.get_network_statistics()
    return stats

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if graph_analyzer else "degraded",
        "service": "Graph Analytics Service",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run("graph_analytics_service:app", host="0.0.0.0", port=8003, reload=True)