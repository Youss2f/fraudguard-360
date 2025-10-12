---
layout: default
title: Architecture
nav_order: 2
description: "FraudGuard 360° system architecture and component overview"
---

# System Architecture
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Architecture Overview

FraudGuard 360° implements a distributed microservices architecture optimized for telecom fraud detection:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dashboard     │────│   API Gateway    │────│  ML Inference   │
│   (React/TS)    │    │   (FastAPI)      │    │   (GraphSAGE)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                       ┌────────▼────────┐      ┌───────▼────────┐
                       │ Stream Processor│      │ Graph Analytics│
                       │  (Flink/Java)   │◄─────┤   (Neo4j)      │
                       └─────────────────┘      └────────────────┘
                                │                        ▲
                       ┌────────▼────────┐               │
                       │   Apache Kafka  │───────────────┘
                       │   (Data Stream) │
                       └─────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    Infrastructure    │
                    │   Neo4j │ Redis      │
                    │ Prometheus │ Grafana │
                    └───────────────────────┘
```

## Core Services

### 1. ML Inference Service (Python)
{: .d-inline-block }

Core ML
{: .label .label-blue }

**Purpose**: Advanced fraud detection with GraphSAGE neural networks

**Technology Stack**:
- PyTorch for deep learning models
- torch-geometric for graph neural networks
- scikit-learn for ensemble methods
- NetworkX for graph operations
- FastAPI for REST API

**Key Features**:
- Real-time GraphSAGE fraud detection model
- Ensemble ML algorithms (XGBoost, LightGBM, Random Forest)
- Velocity pattern analysis
- Neo4j graph integration
- <5 second detection latency

**Endpoints**:
- `POST /predict` - Fraud prediction
- `GET /model/status` - Model health
- `GET /metrics` - Performance metrics

### 2. Stream Processing Service (Java)
{: .d-inline-block }

Stream Processing
{: .label .label-yellow }

**Purpose**: Real-time telecom transaction processing with <5 second detection

**Technology Stack**:
- Apache Flink 1.18 for stream processing
- Kafka connectors for data ingestion
- RocksDB for state management
- Jackson for JSON processing

**Key Features**:
- Real-time transaction stream processing
- Velocity checks and pattern detection
- ML model integration
- Fraud alert generation
- Optimized for telecom data patterns

**Processing Flow**:
1. Consume transactions from Kafka
2. Enrich with historical data
3. Apply velocity checks
4. Score with ML models
5. Generate alerts for suspicious activity

### 3. Graph Analytics Service (Python)
{: .d-inline-block }

Graph Analysis
{: .label .label-green }

**Purpose**: Neo4j-powered fraud network analysis

**Technology Stack**:
- Neo4j for graph database operations
- NetworkX for graph algorithms
- FastAPI for REST API
- Pandas for data manipulation

**Key Features**:
- Fraud ring detection in telecom networks
- Community detection algorithms
- Relationship mapping and analysis
- Network pattern visualization
- Real-time graph updates

**Endpoints**:
- `POST /analyze/network` - Network analysis
- `GET /risk/profile/{id}` - Risk profiling
- `GET /community/detect` - Community detection

### 4. Risk Scoring Service (Python)
{: .d-inline-block }

Risk Assessment
{: .label .label-red }

**Purpose**: Advanced risk assessment algorithms for telecom fraud

**Technology Stack**:
- NumPy and Pandas for numerical computation
- scikit-learn for statistical models
- FastAPI for REST API

**Key Features**:
- 15 distinct risk factors analysis
- Behavioral pattern recognition
- Real-time risk scoring (0-1000 scale)
- Statistical anomaly detection
- Telecom-specific risk evaluation

### 5. API Gateway (Python)
{: .d-inline-block }

Gateway
{: .label .label-purple }

**Purpose**: Unified service orchestration for Sentinel V2 integration

**Technology Stack**:
- FastAPI for high-performance API
- Redis for caching and session management
- JWT for authentication
- Prometheus for metrics

**Key Features**:
- Rate limiting and throttling
- Authentication and authorization
- Service aggregation and routing
- Real-time WebSocket support
- Health monitoring and metrics

### 6. Frontend Dashboard (React TypeScript)
{: .d-inline-block }

UI/UX
{: .label .label-grey-dk-000 }

**Purpose**: Essential fraud monitoring interface

**Technology Stack**:
- React 18 with TypeScript
- Material-UI for components
- Chart.js for visualizations
- WebSocket for real-time updates

**Key Features**:
- Real-time fraud metrics display
- Transaction monitoring table
- Risk score visualization
- Alert management interface
- Professional dark theme

## Data Infrastructure

### Primary Database - Neo4j
{: .d-inline-block }

Graph DB
{: .label .label-blue }

**Purpose**: Relationship analysis and fraud networks

**Features**:
- Graph-based fraud pattern detection
- Network relationship mapping
- Community detection algorithms
- Real-time graph updates

### Caching Layer - Redis
{: .d-inline-block }

Cache
{: .label .label-red }

**Purpose**: High-speed data caching and session management

**Features**:
- Session storage for JWT tokens
- Real-time data caching
- Pub/Sub for notifications
- Performance optimization

### Message Streaming - Apache Kafka
{: .d-inline-block }

Streaming
{: .label .label-yellow }

**Purpose**: Event streaming and microservices communication

**Features**:
- Real-time transaction ingestion
- Event-driven architecture
- Service decoupling
- Fault-tolerant messaging

### Configuration Storage - PostgreSQL
{: .d-inline-block }

SQL DB
{: .label .label-green }

**Purpose**: Configuration and audit logs

**Features**:
- ACID-compliant transactions
- Configuration management
- Audit trail storage
- Backup and recovery

## Technology Distribution

The system maintains an optimal language distribution for telecom fraud detection:

- **Python (53.1%)**: Core ML fraud detection services
- **Java (35.7%)**: High-performance stream processing
- **TypeScript/JS (11.2%)**: Minimal dashboard interface

This distribution ensures:
- ML-focused Python backend for advanced fraud detection
- High-performance Java streaming for real-time processing
- Lightweight frontend for essential monitoring

## Security Architecture

### Authentication & Authorization
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- API key management for service-to-service communication

### Data Protection
- TLS 1.3 encryption for all communications
- Encrypted data at rest
- PII masking and anonymization
- Comprehensive audit logging

### Network Security
- Container isolation with Docker
- Network segmentation
- Rate limiting and DDoS protection
- Vulnerability scanning with Trivy

## Monitoring & Observability

### Metrics Collection
- Prometheus for metrics aggregation
- Custom business metrics for fraud detection
- Performance monitoring across all services

### Visualization
- Grafana dashboards for real-time monitoring
- Fraud detection metrics visualization
- System health and performance dashboards

### Alerting
- Real-time alerts for fraud patterns
- System health notifications
- Performance threshold alerts

---

## Integration with Sentinel V2 Platform

FraudGuard 360° is designed for seamless integration with the Sentinel V2 platform:

- **API Compatibility**: RESTful APIs following OpenAPI specifications
- **Event-Driven Architecture**: Kafka-based messaging for real-time updates
- **Containerized Deployment**: Docker containers for easy deployment
- **Monitoring Integration**: Prometheus metrics for platform monitoring
- **Security Compliance**: Enterprise-grade security standards