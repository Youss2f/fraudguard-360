# FraudGuard 360° - Telecom Fraud Detection System

[![CI/CD Pipeline](https://github.com/Youss2f/fraudguard-360/actions/workflows/main.yml/badge.svg)](https://github.com/Youss2f/fraudguard-360/actions/workflows/main.yml)
[![Security Scan](https://img.shields.io/badge/Security-OWASP%20ZAP-green)](https://github.com/Youss2f/fraudguard-360/security)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Performance](https://img.shields.io/badge/Detection-%3C5s-success)](https://github.com/Youss2f/fraudguard-360)
[![Language Distribution](https://img.shields.io/badge/Python-53.1%25-blue)](https://github.com/Youss2f/fraudguard-360)
[![Architecture](https://img.shields.io/badge/Architecture-Telecom%20Microservices-brightgreen)](https://github.com/Youss2f/fraudguard-360)

> 🎯 **Telecom Fraud Detection Prototype**: Developed for Sentinel V2 platform at Ezi Data EMEA R&D, reducing detection latency from 24 hours to <5 seconds.

## 🚀 Quick Start

```bash
# Start all services
docker-compose up -d

# Verify services
curl http://localhost:8000/health  # API Gateway
curl http://localhost:8001/health  # ML Inference Service
curl http://localhost:8002/health  # Risk Scoring Service  
curl http://localhost:8003/health  # Graph Analytics Service

# Access dashboard
open http://localhost:3000
```

## 🏗️ System Architecture

### Technology Distribution
| Language | Files | Percentage | Purpose |
|----------|-------|------------|---------|
| **Python** | **52** | **53.1%** | **Core ML fraud detection services** |
| **Java** | **35** | **35.7%** | **High-performance stream processing** |
| **TypeScript/JS** | **11** | **11.2%** | **Minimal dashboard interface** |
| **Total** | **98** | **100%** | **Telecom fraud detection architecture** |

### Core Services Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React         │    │   API Gateway    │    │  ML Inference   │
│   Dashboard     │───▶│   (FastAPI)      │───▶│  Service        │
│   (11.2% TS/JS) │    │                  │    │  (GraphSAGE)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                       ┌────────▼────────┐      ┌───────▼────────┐
                       │ Stream          │      │ Risk Scoring   │
                       │ Processing      │◄─────┤ Service        │
                       │ (35.7% Java)    │      │ (Python ML)    │
                       └─────────────────┘      └────────────────┘
                                │                        ▲
                       ┌────────▼────────┐               │
                       │   Apache Kafka  │───────────────┘
                       │   (Real-time)   │
                       └─────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Graph Analytics Svc   │
                    │ Neo4j + NetworkX      │
                    │ (53.1% Python)        │
                    └───────────────────────┘
```

## 📊 Key System Features

### ✅ **Telecom Fraud Detection**
- **Optimized Architecture**: Proper microservices structure for telecom fraud detection ✅
- **Reduced Detection Time**: From 24 hours to <5 seconds ✅  
- **Streamlined Implementation**: Essential components for fraud pattern recognition ✅
- **Production Features**: Real-time processing, ML models, graph analytics ✅

### ✅ **Performance Specifications**
- **Fraud Detection Latency**: <5 seconds for telecom transaction analysis
- **Stream Processing**: Optimized for telecom data patterns
- **API Response Time**: Sub-second for standard requests
- **Graph Queries**: Efficient network analysis for fraud patterns

### ✅ **System Capabilities**
- **Telecom Fraud Detection**: GraphSAGE neural networks with PyTorch
- **Stream Processing**: Apache Flink 1.18 with Java 11
- **Scalable Microservices**: Docker + Kubernetes ready for Sentinel V2
- **Comprehensive Monitoring**: Prometheus + Grafana integration

## 🔧 Technology Stack

### Machine Learning Backend (Python 53.1%)
```python
# Core ML Services
torch>=2.0.0           # GraphSAGE neural networks
scikit-learn>=1.3.0    # Traditional ML algorithms  
networkx>=3.1          # Graph analysis
neo4j>=5.8.0           # Graph database
fastapi>=0.100.0       # High-performance APIs
redis>=4.5.0           # High-speed caching
```

### Stream Processing (Java 35.7%)
```xml
<!-- Apache Flink Ecosystem -->
<flink.version>1.18.0</flink.version>
<kafka.version>3.4.0</kafka.version>
<jackson.version>2.15.2</jackson.version>
```

### Minimal Frontend (TypeScript/JS 11.2%)
```json
{
  "react": "^18.2.0",
  "typescript": "^4.9.5",
  "dependencies": "minimal-essential-only"
}
```

## 🎯 Core Services Overview

### 1. **ML Inference Service** (Port 8001)
- **Purpose**: Advanced fraud detection with GraphSAGE neural networks
- **Features**: Real-time ML inference, ensemble algorithms, velocity analysis
- **Technology**: PyTorch, torch-geometric, scikit-learn, NetworkX

### 2. **Risk Scoring Service** (Port 8002)
- **Purpose**: Advanced risk assessment algorithms  
- **Features**: 15 risk factors, behavioral analysis, statistical modeling
- **Technology**: NumPy, pandas, scikit-learn, statistical models

### 3. **Graph Analytics Service** (Port 8003)
- **Purpose**: Neo4j-powered fraud network analysis
- **Features**: Fraud ring detection, community analysis, network patterns
- **Technology**: Neo4j, NetworkX, graph algorithms

### 4. **Stream Processing Service**
- **Purpose**: Real-time telecom transaction processing (<5 second detection)
- **Features**: Telecom data processing, pattern detection, ML integration
- **Technology**: Apache Flink 1.18, Kafka connectors, RocksDB state

### 5. **API Gateway** (Port 8000)
- **Purpose**: Unified service orchestration
- **Features**: Rate limiting, authentication, service aggregation
- **Technology**: FastAPI, Redis caching, JWT authentication

## 🚀 Development & Deployment

### Local Development
```bash
# Start infrastructure
docker-compose up -d kafka neo4j redis postgres

# Run services locally
cd core-ml-service && python -m uvicorn ml_service_api:app --reload --port 8001
cd risk-scoring-service && python -m uvicorn risk_scoring_service:app --reload --port 8002  
cd graph-analytics-service && python -m uvicorn graph_analytics_service:app --reload --port 8003
cd api-gateway && python -m uvicorn app.main:app --reload --port 8000
cd frontend && npm start
```

### Production Deployment
```bash
# Docker Compose
docker-compose -f docker-compose.production.yml up -d

# Kubernetes
kubectl apply -f k8s/production/
```

## 🛡️ Security & Compliance

- **Authentication**: JWT tokens with configurable expiration
- **Rate Limiting**: 2000 requests/minute per client
- **Data Protection**: PII masking and encryption
- **Audit Logging**: Comprehensive transaction and access logs
- **Vulnerability Scanning**: Automated security scanning in CI/CD

## 📈 Monitoring & Observability

- **Metrics**: Prometheus with business and technical KPIs
- **Dashboards**: Grafana with pre-configured fraud detection dashboards  
- **Alerting**: Real-time alerts for fraud patterns and system health
- **Logging**: Structured logging with correlation IDs

## 📖 Documentation

- **[System Overview](RESTRUCTURE_SUMMARY.md)**: Detailed technical architecture overview
- **[Deployment Guide](DEPLOYMENT.md)**: Production deployment instructions
- **[API Documentation](http://localhost:8000/docs)**: Interactive API explorer
- **[Architecture Overview](infrastructure/README.md)**: System design details

## 🏆 System Overview

**FraudGuard 360°** is a telecom fraud detection system prototype developed for the Sentinel V2 platform at Ezi Data EMEA R&D, reducing detection latency from 24 hours to under 5 seconds.

> See **[RESTRUCTURE_SUMMARY.md](RESTRUCTURE_SUMMARY.md)** for complete system details and technical implementation overview.