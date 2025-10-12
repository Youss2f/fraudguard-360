---
layout: default
title: Getting Started
nav_order: 3
description: "Complete setup and installation guide for FraudGuard 360°"
---

# Getting Started
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Before setting up FraudGuard 360°, ensure you have the following installed:

### Required Software

| Component | Version | Purpose |
|-----------|---------|---------|
| **Docker** | 24.0+ | Container runtime |
| **Docker Compose** | 2.20+ | Multi-container orchestration |
| **Node.js** | 18+ | Frontend development |
| **Python** | 3.11+ | Service development |
| **Java** | 17+ | Stream processing service |
| **Git** | 2.30+ | Version control |

### System Requirements

- **RAM**: Minimum 8GB, Recommended 16GB
- **CPU**: Minimum 4 cores, Recommended 8 cores
- **Storage**: Minimum 10GB free space
- **Network**: Internet connection for dependency downloads

## Installation Methods

### Method 1: Docker Compose (Recommended)
{: .d-inline-block }

Production Ready
{: .label .label-green }

This method sets up the complete system with all services and infrastructure.

#### 1. Clone the Repository

```bash
git clone https://github.com/Youss2f/fraudguard-360.git
cd fraudguard-360
```

#### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional for development)
# nano .env
```

#### 3. Start All Services

```bash
# Start all services with infrastructure
docker-compose up -d

# Check service health
docker-compose ps
```

#### 4. Verify Installation

```bash
# Check service health endpoints
curl http://localhost:8000/health    # API Gateway
curl http://localhost:8001/health    # ML Inference Service
curl http://localhost:8002/health    # Risk Scoring Service
curl http://localhost:8003/health    # Graph Analytics Service
```

### Method 2: Development Setup
{: .d-inline-block }

Development
{: .label .label-blue }

For developers who want to run services locally for development.

#### 1. Start Infrastructure Only

```bash
# Start only infrastructure services
docker-compose up -d kafka neo4j redis postgres prometheus grafana
```

#### 2. Set Up Python Services

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies for each Python service
cd core-ml-service
pip install -r requirements.txt

cd ../risk-scoring-service
pip install -r requirements.txt

cd ../graph-analytics-service
pip install -r requirements.txt

cd ../api-gateway
pip install -r requirements.txt
```

#### 3. Set Up Java Stream Processing

```bash
cd stream-processor-flink
mvn clean install
```

#### 4. Set Up Frontend

```bash
cd frontend
npm install
```

#### 5. Run Services Locally

```bash
# Terminal 1 - ML Inference Service
cd core-ml-service
python -m uvicorn ml_service_api:app --reload --port 8001

# Terminal 2 - Risk Scoring Service
cd risk-scoring-service
python -m uvicorn risk_scoring_service:app --reload --port 8002

# Terminal 3 - Graph Analytics Service
cd graph-analytics-service
python -m uvicorn graph_analytics_service:app --reload --port 8003

# Terminal 4 - API Gateway
cd api-gateway
python -m uvicorn app.main:app --reload --port 8000

# Terminal 5 - Frontend
cd frontend
npm start

# Terminal 6 - Flink Stream Processor
cd stream-processor-flink
mvn exec:java -Dexec.mainClass="com.fraudguard360.FraudDetectionStreamProcessor"
```

## Access Points

Once the system is running, you can access the following endpoints:

### User Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend Dashboard** | [http://localhost:3000](http://localhost:3000) | - |
| **Grafana Monitoring** | [http://localhost:3001](http://localhost:3001) | admin/admin |
| **Neo4j Browser** | [http://localhost:7474](http://localhost:7474) | neo4j/fraudguard360 |

### API Endpoints

| Service | URL | Documentation |
|---------|-----|---------------|
| **API Gateway** | [http://localhost:8000](http://localhost:8000) | [/docs](http://localhost:8000/docs) |
| **ML Inference** | [http://localhost:8001](http://localhost:8001) | [/docs](http://localhost:8001/docs) |
| **Risk Scoring** | [http://localhost:8002](http://localhost:8002) | [/docs](http://localhost:8002/docs) |
| **Graph Analytics** | [http://localhost:8003](http://localhost:8003) | [/docs](http://localhost:8003/docs) |

### Monitoring & Metrics

| Service | URL | Purpose |
|---------|-----|---------|
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Metrics collection |
| **Grafana** | [http://localhost:3001](http://localhost:3001) | Dashboards & visualization |

## Basic Usage

### 1. Analyze a Transaction

```bash
curl -X POST http://localhost:8000/api/v1/fraud/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_123456",
    "amount": 1500.00,
    "currency": "USD",
    "merchant_id": "merchant_789",
    "customer_id": "customer_456",
    "timestamp": "2025-10-10T10:30:00Z",
    "location": {
      "country": "US",
      "city": "New York"
    }
  }'
```

**Expected Response**:
```json
{
  "transaction_id": "txn_123456",
  "fraud_score": 0.75,
  "risk_level": "HIGH",
  "prediction": "FRAUD",
  "confidence": 0.89,
  "processing_time_ms": 245,
  "risk_factors": [
    "unusual_amount",
    "velocity_pattern",
    "location_anomaly"
  ],
  "recommendation": "BLOCK_TRANSACTION"
}
```

### 2. Batch Analysis

```bash
curl -X POST http://localhost:8000/api/v1/fraud/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "txn_001",
        "amount": 500.00,
        "customer_id": "customer_123"
      },
      {
        "transaction_id": "txn_002", 
        "amount": 2500.00,
        "customer_id": "customer_456"
      }
    ]
  }'
```

### 3. Get System Metrics

```bash
curl http://localhost:8000/api/v1/metrics
```

### 4. Graph Network Analysis

```bash
curl -X POST http://localhost:8003/analyze/network \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "customer_456",
    "depth": 2,
    "min_transaction_amount": 1000
  }'
```

## Configuration

### Environment Variables

Key configuration options in `.env` file:

```bash
# Database Configuration
POSTGRES_URL=postgresql://fraud_user:fraud_pass@localhost:5432/fraudguard
REDIS_URL=redis://localhost:6379/0
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=fraudguard360

# Message Queue
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API Configuration
API_RATE_LIMIT=2000  # requests per minute
CORS_ORIGINS=http://localhost:3000

# ML Model Configuration
ML_MODEL_PATH=/app/models/fraud_detection.pkl
GRAPH_MODEL_PATH=/app/models/graphsage_model.pt

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
```

### Service Ports

| Service | Port | Protocol |
|---------|------|----------|
| Frontend | 3000 | HTTP |
| API Gateway | 8000 | HTTP |
| ML Inference | 8001 | HTTP |
| Risk Scoring | 8002 | HTTP |
| Graph Analytics | 8003 | HTTP |
| PostgreSQL | 5432 | TCP |
| Redis | 6379 | TCP |
| Neo4j | 7687, 7474 | Bolt, HTTP |
| Kafka | 9092 | TCP |
| Prometheus | 9090 | HTTP |
| Grafana | 3001 | HTTP |

## Testing the Installation

### Health Checks

Verify all services are running correctly:

```bash
#!/bin/bash
echo "=== FraudGuard 360° Health Check ==="

services=("8000:API Gateway" "8001:ML Inference" "8002:Risk Scoring" "8003:Graph Analytics")

for service in "${services[@]}"; do
    port=$(echo $service | cut -d: -f1)
    name=$(echo $service | cut -d: -f2)
    
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health)
    
    if [ $response -eq 200 ]; then
        echo "✅ $name: Healthy"
    else
        echo "❌ $name: Unhealthy (HTTP $response)"
    fi
done
```

### Integration Test

Run a complete end-to-end test:

```bash
# Test fraud detection pipeline
curl -X POST http://localhost:8000/api/v1/fraud/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "test_txn_001",
    "amount": 5000.00,
    "currency": "USD",
    "merchant_id": "test_merchant",
    "customer_id": "test_customer",
    "timestamp": "2025-10-10T15:30:00Z"
  }' | jq .
```

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep :8000

# Stop conflicting services
sudo systemctl stop apache2  # If using port 8000
```

#### Docker Issues
```bash
# Restart Docker
sudo systemctl restart docker

# Clean up containers
docker-compose down
docker system prune -f
```

#### Memory Issues
```bash
# Check available memory
free -h

# Reduce Docker memory usage
docker-compose down
docker system prune -a -f
```

#### Service Dependencies
```bash
# Check service logs
docker-compose logs api-gateway
docker-compose logs ml-service

# Restart specific service
docker-compose restart api-gateway
```

### Log Analysis

View service logs for debugging:

```bash
# View all service logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api-gateway
docker-compose logs -f ml-service

# View last 100 lines
docker-compose logs --tail=100 api-gateway
```

## Next Steps

1. **Explore the Dashboard**: Visit [http://localhost:3000](http://localhost:3000) to see the fraud detection interface
2. **Read API Documentation**: Check out the interactive API docs at [http://localhost:8000/docs](http://localhost:8000/docs)
3. **Review Architecture**: Understand the system design in our [Architecture Guide](architecture.html)
4. **Performance Testing**: Run performance tests with our [Performance Guide](performance.html)
5. **Deployment**: Deploy to production using our [Deployment Guide](deployment.html)

---

Need help? Check our [troubleshooting guide](contributing.html#troubleshooting) or [open an issue](https://github.com/Youss2f/fraudguard-360/issues) on GitHub.