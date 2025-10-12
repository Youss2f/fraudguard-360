-- PostgreSQL schema setup for FraudGuard 360
-- This script creates tables for fraud cases, user metadata, and operational data

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS fraud_detection;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Set search path
SET search_path TO fraud_detection, public;

-- Users table (metadata not covered by Neo4j)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    account_status VARCHAR(20) DEFAULT 'active',
    subscription_plan VARCHAR(50),
    billing_address JSONB,
    preferences JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Fraud cases table
CREATE TABLE IF NOT EXISTS fraud_cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_number VARCHAR(50) UNIQUE NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    fraud_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'confirmed', 'false_positive', 'closed')),
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    risk_score DECIMAL(5,4) CHECK (risk_score >= 0 AND risk_score <= 1),
    amount_at_risk DECIMAL(15,2),
    description TEXT,
    evidence JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    assigned_to VARCHAR(100),
    resolution TEXT,
    resolution_date TIMESTAMP WITH TIME ZONE,
    false_positive_reason TEXT,
    tags TEXT[],
    priority INTEGER DEFAULT 3 CHECK (priority >= 1 AND priority <= 5)
);

-- Fraud alerts table (real-time alerts from ML system)
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(100) UNIQUE NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    fraud_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(5,4),
    risk_score DECIMAL(5,4),
    description TEXT,
    evidence JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    escalated_to_case UUID REFERENCES fraud_cases(id),
    source_system VARCHAR(50) DEFAULT 'ml_service',
    raw_data JSONB
);

-- Investigations table
CREATE TABLE IF NOT EXISTS investigations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID NOT NULL REFERENCES fraud_cases(id) ON DELETE CASCADE,
    investigator_id VARCHAR(100) NOT NULL,
    investigation_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    findings TEXT,
    recommendations TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    priority INTEGER DEFAULT 3,
    metadata JSONB DEFAULT '{}'
);

-- Investigation actions table
CREATE TABLE IF NOT EXISTS investigation_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    investigation_id UUID NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    action_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    taken_by VARCHAR(100) NOT NULL,
    taken_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB DEFAULT '{}',
    outcome TEXT
);

-- ML model predictions log
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL,
    fraud_probability DECIMAL(5,4),
    anomaly_score DECIMAL(5,4),
    final_risk_score DECIMAL(5,4),
    risk_level VARCHAR(20),
    features JSONB,
    prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER,
    model_confidence DECIMAL(5,4)
);

-- Fraud patterns table
CREATE TABLE IF NOT EXISTS fraud_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id VARCHAR(100) UNIQUE NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    description TEXT,
    detection_rules JSONB,
    confidence_threshold DECIMAL(5,4) DEFAULT 0.7,
    severity_mapping JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    tags TEXT[]
);

-- Pattern matches table
CREATE TABLE IF NOT EXISTS pattern_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_id UUID NOT NULL REFERENCES fraud_patterns(id),
    user_id VARCHAR(50) NOT NULL,
    match_score DECIMAL(5,4),
    matched_features JSONB,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    alert_generated BOOLEAN DEFAULT false,
    alert_id UUID REFERENCES fraud_alerts(id)
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(20) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6),
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    evaluation_dataset VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

-- Audit schema tables
CREATE TABLE IF NOT EXISTS audit.fraud_case_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID NOT NULL,
    field_name VARCHAR(100) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    changed_by VARCHAR(100) NOT NULL,
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    change_reason TEXT
);

-- Analytics schema tables
CREATE TABLE IF NOT EXISTS analytics.daily_fraud_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    total_alerts INTEGER DEFAULT 0,
    critical_alerts INTEGER DEFAULT 0,
    high_alerts INTEGER DEFAULT 0,
    confirmed_frauds INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    total_amount_at_risk DECIMAL(15,2) DEFAULT 0,
    avg_detection_time_hours DECIMAL(8,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_fraud_cases_user_id ON fraud_cases(user_id);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_status ON fraud_cases(status);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_severity ON fraud_cases(severity);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_created_at ON fraud_cases(created_at);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_fraud_type ON fraud_cases(fraud_type);

CREATE INDEX IF NOT EXISTS idx_fraud_alerts_user_id ON fraud_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_severity ON fraud_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_created_at ON fraud_alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts(status);

CREATE INDEX IF NOT EXISTS idx_investigations_case_id ON investigations(case_id);
CREATE INDEX IF NOT EXISTS idx_investigations_investigator ON investigations(investigator_id);
CREATE INDEX IF NOT EXISTS idx_investigations_status ON investigations(status);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_user_id ON ml_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_timestamp ON ml_predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_risk_score ON ml_predictions(final_risk_score);

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_fraud_cases_evidence_gin ON fraud_cases USING gin(evidence);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_evidence_gin ON fraud_alerts USING gin(evidence);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_features_gin ON ml_predictions USING gin(features);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_fraud_cases_user_status ON fraud_cases(user_id, status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_user_severity ON fraud_alerts(user_id, severity);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_severity_created ON fraud_cases(severity, created_at DESC);

-- Full text search indexes
CREATE INDEX IF NOT EXISTS idx_fraud_cases_description_fts ON fraud_cases USING gin(to_tsvector('english', description));
CREATE INDEX IF NOT EXISTS idx_investigations_findings_fts ON investigations USING gin(to_tsvector('english', findings));

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fraud_cases_updated_at BEFORE UPDATE ON fraud_cases 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for fraud case audit logging
CREATE OR REPLACE FUNCTION log_fraud_case_changes()
RETURNS TRIGGER AS $$
BEGIN
    -- Log status changes
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO audit.fraud_case_history (case_id, field_name, old_value, new_value, changed_by)
        VALUES (NEW.id, 'status', OLD.status, NEW.status, current_user);
    END IF;
    
    -- Log severity changes
    IF OLD.severity IS DISTINCT FROM NEW.severity THEN
        INSERT INTO audit.fraud_case_history (case_id, field_name, old_value, new_value, changed_by)
        VALUES (NEW.id, 'severity', OLD.severity, NEW.severity, current_user);
    END IF;
    
    -- Log assignment changes
    IF OLD.assigned_to IS DISTINCT FROM NEW.assigned_to THEN
        INSERT INTO audit.fraud_case_history (case_id, field_name, old_value, new_value, changed_by)
        VALUES (NEW.id, 'assigned_to', OLD.assigned_to, NEW.assigned_to, current_user);
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER audit_fraud_case_changes 
    AFTER UPDATE ON fraud_cases
    FOR EACH ROW EXECUTE FUNCTION log_fraud_case_changes();

-- Insert sample data for development
INSERT INTO users (user_id, email, account_status, subscription_plan) VALUES
    ('user_001', 'john.doe@example.com', 'active', 'premium'),
    ('user_002', 'jane.smith@example.com', 'active', 'basic'),
    ('user_003', 'suspicious.user@example.com', 'active', 'unlimited')
ON CONFLICT (user_id) DO NOTHING;

INSERT INTO fraud_patterns (pattern_id, pattern_type, description, detection_rules, confidence_threshold) VALUES
    ('sim_box_pattern_001', 'sim_box', 'High volume short international calls', 
     '{"min_international_calls": 50, "max_avg_duration": 60, "min_call_frequency": 20}', 0.85),
    ('velocity_pattern_001', 'velocity_anomaly', 'Unusual call frequency spike', 
     '{"call_frequency_threshold": 30, "time_window_minutes": 60}', 0.75),
    ('premium_rate_pattern_001', 'premium_rate_fraud', 'Multiple calls to premium numbers', 
     '{"min_premium_calls": 5, "min_cost_per_call": 5.0}', 0.80)
ON CONFLICT (pattern_id) DO NOTHING;

-- Insert a sample fraud case
INSERT INTO fraud_cases (
    case_number, user_id, fraud_type, severity, confidence_score, risk_score,
    amount_at_risk, description, evidence, assigned_to
) VALUES (
    'CASE-2024-001', 'user_003', 'sim_box', 'high', 0.85, 0.90,
    1500.00, 'High volume of short international calls detected',
    '{"international_calls": 150, "avg_duration": 45, "call_frequency": 25, "suspicious_destinations": ["IN", "PK", "NG"]}',
    'fraud_analyst_001'
) ON CONFLICT (case_number) DO NOTHING;

-- Insert sample fraud alert
INSERT INTO fraud_alerts (
    alert_id, user_id, fraud_type, severity, confidence_score, risk_score,
    description, evidence
) VALUES (
    'alert_001', 'user_003', 'sim_box', 'high', 0.85, 0.90,
    'SIM box fraud pattern detected: High volume international calls',
    '{"call_frequency": 25.5, "international_calls": 150, "premium_rate_calls": 0, "night_calls": 45}'
) ON CONFLICT (alert_id) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW active_fraud_cases AS
SELECT 
    fc.*,
    u.email as user_email,
    COUNT(fa.id) as related_alerts_count,
    MAX(fa.created_at) as latest_alert_time
FROM fraud_cases fc
LEFT JOIN users u ON fc.user_id = u.user_id
LEFT JOIN fraud_alerts fa ON fc.user_id = fa.user_id 
    AND fa.created_at >= fc.created_at - INTERVAL '7 days'
WHERE fc.status IN ('open', 'investigating')
GROUP BY fc.id, u.email;

CREATE OR REPLACE VIEW high_risk_alerts AS
SELECT 
    fa.*,
    u.email as user_email,
    fc.case_number as related_case
FROM fraud_alerts fa
LEFT JOIN users u ON fa.user_id = u.user_id
LEFT JOIN fraud_cases fc ON fa.escalated_to_case = fc.id
WHERE fa.severity IN ('high', 'critical') 
    AND fa.status = 'open'
ORDER BY fa.risk_score DESC, fa.created_at DESC;

CREATE OR REPLACE VIEW fraud_detection_metrics AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_alerts,
    COUNT(*) FILTER (WHERE severity = 'critical') as critical_alerts,
    COUNT(*) FILTER (WHERE severity = 'high') as high_alerts,
    COUNT(*) FILTER (WHERE severity = 'medium') as medium_alerts,
    COUNT(*) FILTER (WHERE severity = 'low') as low_alerts,
    AVG(confidence_score) as avg_confidence,
    AVG(risk_score) as avg_risk_score
FROM fraud_alerts 
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Create stored procedures for common operations
CREATE OR REPLACE FUNCTION create_fraud_case(
    p_user_id VARCHAR(50),
    p_fraud_type VARCHAR(50),
    p_severity VARCHAR(20),
    p_confidence_score DECIMAL(5,4),
    p_risk_score DECIMAL(5,4),
    p_description TEXT,
    p_evidence JSONB DEFAULT '{}',
    p_assigned_to VARCHAR(100) DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    case_id UUID;
    case_number VARCHAR(50);
BEGIN
    -- Generate case number
    SELECT 'CASE-' || TO_CHAR(CURRENT_DATE, 'YYYY') || '-' || 
           LPAD((COUNT(*) + 1)::TEXT, 3, '0')
    INTO case_number
    FROM fraud_cases 
    WHERE created_at >= DATE_TRUNC('year', CURRENT_DATE);
    
    -- Insert fraud case
    INSERT INTO fraud_cases (
        case_number, user_id, fraud_type, severity, confidence_score, 
        risk_score, description, evidence, assigned_to
    ) VALUES (
        case_number, p_user_id, p_fraud_type, p_severity, p_confidence_score,
        p_risk_score, p_description, p_evidence, p_assigned_to
    ) RETURNING id INTO case_id;
    
    RETURN case_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION escalate_alert_to_case(
    p_alert_id UUID
) RETURNS UUID AS $$
DECLARE
    alert_record fraud_alerts%ROWTYPE;
    case_id UUID;
BEGIN
    -- Get alert details
    SELECT * INTO alert_record FROM fraud_alerts WHERE id = p_alert_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Alert not found';
    END IF;
    
    -- Create fraud case
    SELECT create_fraud_case(
        alert_record.user_id,
        alert_record.fraud_type,
        alert_record.severity,
        alert_record.confidence_score,
        alert_record.risk_score,
        'Escalated from alert: ' || alert_record.description,
        alert_record.evidence
    ) INTO case_id;
    
    -- Update alert
    UPDATE fraud_alerts 
    SET escalated_to_case = case_id, status = 'escalated'
    WHERE id = p_alert_id;
    
    RETURN case_id;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your environment)
-- GRANT USAGE ON SCHEMA fraud_detection TO fraud_detection_user;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA fraud_detection TO fraud_detection_user;
-- GRANT USAGE ON ALL SEQUENCES IN SCHEMA fraud_detection TO fraud_detection_user;

SELECT 'PostgreSQL schema setup completed successfully' as status;