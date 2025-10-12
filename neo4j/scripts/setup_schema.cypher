// Neo4j schema setup for FraudGuard 360
// Run this script to create constraints, indexes, and initial graph structure

// Create constraints for unique identifiers
CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT device_imei_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.imei IS UNIQUE;
CREATE CONSTRAINT call_id_unique IF NOT EXISTS FOR (c:Call) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT alert_id_unique IF NOT EXISTS FOR (a:FraudAlert) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT location_name_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE;

// Create indexes for performance
CREATE INDEX user_phone_index IF NOT EXISTS FOR (u:User) ON (u.phone_number);
CREATE INDEX call_timestamp_index IF NOT EXISTS FOR (c:Call) ON (c.timestamp);
CREATE INDEX alert_severity_index IF NOT EXISTS FOR (a:FraudAlert) ON (a.severity);
CREATE INDEX alert_timestamp_index IF NOT EXISTS FOR (a:FraudAlert) ON (a.timestamp);
CREATE INDEX user_risk_score_index IF NOT EXISTS FOR (u:User) ON (u.risk_score);

// Create composite indexes for common queries
CREATE INDEX user_status_risk_index IF NOT EXISTS FOR (u:User) ON (u.account_status, u.risk_score);
CREATE INDEX call_type_cost_index IF NOT EXISTS FOR (c:Call) ON (c.type, c.cost);

// Initialize some sample data for development (optional)
// Create sample users
MERGE (u1:User {
    id: 'user_001',
    phone_number: '+1234567890',
    name: 'John Doe',
    email: 'john@example.com',
    registration_date: datetime('2023-01-15T00:00:00Z'),
    account_status: 'active',
    plan_type: 'premium',
    risk_score: 0.2,
    call_count: 150,
    sms_count: 80,
    total_duration: 12000,
    total_cost: 45.50,
    international_calls: 5,
    premium_rate_calls: 0,
    unique_callees: 25,
    account_age_days: 600,
    plan_cost: 50.0
});

MERGE (u2:User {
    id: 'user_002',
    phone_number: '+1234567891',
    name: 'Jane Smith',
    email: 'jane@example.com',
    registration_date: datetime('2024-06-01T00:00:00Z'),
    account_status: 'active',
    plan_type: 'basic',
    risk_score: 0.1,
    call_count: 80,
    sms_count: 120,
    total_duration: 6000,
    total_cost: 25.30,
    international_calls: 2,
    premium_rate_calls: 0,
    unique_callees: 15,
    account_age_days: 120,
    plan_cost: 30.0
});

MERGE (u3:User {
    id: 'user_003',
    phone_number: '+1234567892',
    name: 'Suspicious User',
    email: 'suspicious@example.com',
    registration_date: datetime('2024-08-01T00:00:00Z'),
    account_status: 'active',
    plan_type: 'unlimited',
    risk_score: 0.85,
    call_count: 1500,
    sms_count: 200,
    total_duration: 45000,
    total_cost: 500.75,
    international_calls: 300,
    premium_rate_calls: 25,
    unique_callees: 500,
    account_age_days: 60,
    plan_cost: 100.0
});

// Create sample devices
MERGE (d1:Device {
    imei: 'IMEI_iPhone_001',
    model: 'iPhone 14',
    manufacturer: 'Apple',
    first_seen: datetime('2023-01-15T00:00:00Z'),
    last_seen: datetime('2024-09-26T00:00:00Z'),
    status: 'active'
});

MERGE (d2:Device {
    imei: 'IMEI_Samsung_001',
    model: 'Galaxy S23',
    manufacturer: 'Samsung',
    first_seen: datetime('2024-06-01T00:00:00Z'),
    last_seen: datetime('2024-09-26T00:00:00Z'),
    status: 'active'
});

MERGE (d3:Device {
    imei: 'IMEI_Suspicious_001',
    model: 'Unknown',
    manufacturer: 'Unknown',
    first_seen: datetime('2024-08-01T00:00:00Z'),
    last_seen: datetime('2024-09-26T00:00:00Z'),
    status: 'flagged'
});

// Create sample locations
MERGE (l1:Location {name: 'New York', country: 'US', latitude: 40.7128, longitude: -74.0060});
MERGE (l2:Location {name: 'Los Angeles', country: 'US', latitude: 34.0522, longitude: -118.2437});
MERGE (l3:Location {name: 'Mumbai', country: 'IN', latitude: 19.0760, longitude: 72.8777});
MERGE (l4:Location {name: 'Lagos', country: 'NG', latitude: 6.5244, longitude: 3.3792});

// Create relationships between users and devices
MERGE (u1)-[:USES_DEVICE {first_used: datetime('2023-01-15T00:00:00Z')}]->(d1);
MERGE (u2)-[:USES_DEVICE {first_used: datetime('2024-06-01T00:00:00Z')}]->(d2);
MERGE (u3)-[:USES_DEVICE {first_used: datetime('2024-08-01T00:00:00Z')}]->(d3);

// Create sample calls
MERGE (c1:Call {
    id: 'call_001',
    type: 'voice',
    startTime: datetime('2024-09-26T10:00:00Z'),
    duration: 180,
    cost: 0.50,
    countryCode: 'US',
    timestamp: datetime('2024-09-26T10:00:00Z')
});

MERGE (c2:Call {
    id: 'call_002',
    type: 'voice',
    startTime: datetime('2024-09-26T10:30:00Z'),
    duration: 45,
    cost: 2.50,
    countryCode: 'IN',
    timestamp: datetime('2024-09-26T10:30:00Z')
});

// Create call relationships
MERGE (u1)-[:MADE_CALL]->(c1);
MERGE (c1)-[:TO]->(u2);
MERGE (u3)-[:MADE_CALL]->(c2);
MERGE (c2)-[:TO {international: true}]->(u1);

// Create location relationships
MERGE (u1)-[:LOCATED_AT {timestamp: datetime('2024-09-26T10:00:00Z')}]->(l1);
MERGE (u2)-[:LOCATED_AT {timestamp: datetime('2024-09-26T10:00:00Z')}]->(l2);
MERGE (u3)-[:LOCATED_AT {timestamp: datetime('2024-09-26T10:30:00Z')}]->(l3);

// Create a fraud alert for the suspicious user
MERGE (alert1:FraudAlert {
    id: 'alert_001',
    fraudType: 'sim_box',
    severity: 'high',
    confidenceScore: 0.85,
    riskScore: 0.90,
    description: 'High volume of short international calls detected',
    status: 'open',
    timestamp: datetime('2024-09-26T11:00:00Z')
});

MERGE (u3)-[:HAS_ALERT]->(alert1);

// Create fraud patterns for analysis
MERGE (pattern1:FraudPattern {
    id: 'pattern_001',
    type: 'velocity_anomaly',
    description: 'Unusual call frequency detected',
    confidence: 0.75,
    timestamp: datetime('2024-09-26T11:00:00Z')
});

MERGE (u3)-[:EXHIBITS_PATTERN]->(pattern1);

// Useful queries for fraud detection:

// Query 1: Find users with high risk scores
// MATCH (u:User) WHERE u.risk_score > 0.7 RETURN u ORDER BY u.risk_score DESC;

// Query 2: Find users with many international calls
// MATCH (u:User)-[:MADE_CALL]->(c:Call) WHERE c.countryCode <> 'US' 
// WITH u, count(c) as intl_calls WHERE intl_calls > 10 
// RETURN u.id, u.name, intl_calls ORDER BY intl_calls DESC;

// Query 3: Find suspicious calling patterns (many short calls)
// MATCH (u:User)-[:MADE_CALL]->(c:Call) WHERE c.duration < 60 
// WITH u, count(c) as short_calls WHERE short_calls > 50 
// RETURN u.id, u.name, short_calls ORDER BY short_calls DESC;

// Query 4: Find users calling the same premium numbers
// MATCH (u1:User)-[:MADE_CALL]->(c1:Call)-[:TO]->(target)
// MATCH (u2:User)-[:MADE_CALL]->(c2:Call)-[:TO]->(target)
// WHERE u1 <> u2 AND c1.cost > 5.0 AND c2.cost > 5.0
// RETURN u1.id, u2.id, target.phone_number, count(*) as shared_calls
// ORDER BY shared_calls DESC;

// Query 5: Network analysis - find connected users
// MATCH path = (u1:User)-[:MADE_CALL|TO*1..3]-(u2:User)
// WHERE u1.id = 'user_003'
// RETURN path LIMIT 20;

// Query 6: Find fraud communities (users with similar patterns)
// MATCH (u1:User)-[:HAS_ALERT]->(a1:FraudAlert)
// MATCH (u2:User)-[:HAS_ALERT]->(a2:FraudAlert)
// WHERE u1 <> u2 AND a1.fraudType = a2.fraudType
// RETURN u1.id, u2.id, a1.fraudType, a1.severity, a2.severity;

RETURN "Neo4j schema setup completed successfully" as status;