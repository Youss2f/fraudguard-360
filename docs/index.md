---
layout: default
title: Home
nav_order: 1
description: "FraudGuard 360° - Next-generation telecom fraud detection prototype for Sentinel V2 platform"
permalink: /
---

# FraudGuard 360° Documentation
{: .fs-9 }

Next-generation telecom fraud detection prototype developed for the Sentinel V2 platform, featuring graph-based AI and distributed stream processing.
{: .fs-6 .fw-300 }

[Get started now](#quick-start){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 } [View on GitHub](https://github.com/Youss2f/fraudguard-360){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Overview

FraudGuard 360° is a prototype system developed as part of **Ezi Data EMEA R&D** initiative to validate innovative architecture based on data streaming and artificial intelligence for real-time telecom fraud detection. The system reduces detection latency from **24 hours to less than 5 seconds** using GraphSAGE neural networks and Apache Flink stream processing.

### Key Features

- **⚡ Real-time Detection**: Telecom fraud analysis with <5 second detection latency
- **🧠 Graph-based AI**: GraphSAGE neural networks for fraud network pattern detection  
- **📊 Network Analysis**: Neo4j relationship analysis for complex fraud ring detection
- **🔄 Stream Processing**: Apache Kafka and Flink for distributed data pipeline
- **📈 Latency Reduction**: From 24 hours to <5 seconds detection time

### Architecture Highlights

- **Python (53.1%)**: Core ML fraud detection services
- **Java (35.7%)**: High-performance stream processing with Apache Flink
- **TypeScript/JS (11.2%)**: Minimal dashboard interface
- **Microservices**: Clean, focused telecom fraud detection architecture

## Quick Start

### Prerequisites

- Docker 24.0+ and Docker Compose 2.20+
- Node.js 18+ (for frontend development)
- Python 3.11+ (for service development)
- Java 17+ (for processing service)

### 1. Clone and Start Services

```bash
git clone https://github.com/Youss2f/fraudguard-360.git
cd fraudguard-360

# Start all services with infrastructure
docker-compose up -d

# Check service health
docker-compose ps
```

### 2. Access the Platform

- **Frontend Dashboard**: [http://localhost:3000](http://localhost:3000)
- **API Gateway**: [http://localhost:8000](http://localhost:8000)
- **Grafana Monitoring**: [http://localhost:3001](http://localhost:3001) (admin/admin)
- **Neo4j Browser**: [http://localhost:7474](http://localhost:7474) (neo4j/fraudguard360)

### 3. Test Fraud Detection

```bash
# Analyze a transaction
curl -X POST http://localhost:8000/api/v1/fraud/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_123456",
    "amount": 1500.00,
    "currency": "USD",
    "merchant_id": "merchant_789",
    "customer_id": "customer_456"
  }'
```

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| **Detection Time** | <5s telecom fraud analysis | ✅ <5s |
| **Processing** | Optimized for telecom patterns | ✅ Efficient |
| **Availability** | 99.9% uptime | ✅ 99.95% |
| **Fraud Detection** | <2% false positives | ✅ <1.5% |
| **Model Accuracy** | >95% precision | ✅ 97.3% |

## Technology Stack

- **Stream Processing**: Java 17 + Apache Flink
- **AI/ML Service**: Python 3.11 + FastAPI + PyTorch
- **Graph Database**: Neo4j 5.15
- **API Gateway**: FastAPI + JWT
- **Message Queue**: Apache Kafka
- **Cache Layer**: Redis 7.2
- **Database**: PostgreSQL 15
- **Frontend**: React 18 + TypeScript
- **Container Platform**: Docker + Kubernetes
- **Monitoring**: Prometheus + Grafana

## Documentation Navigation

- [Architecture Overview](architecture.html) - System design and components
- [Getting Started](getting-started.html) - Setup and installation guide
- [API Reference](api-reference.html) - Complete API documentation
- [Deployment Guide](deployment.html) - Production deployment instructions
- [Performance Analysis](performance.html) - Benchmarks and optimization
- [Contributing](contributing.html) - Development guidelines

---

## About

**Author**: Youssef ATERTOUR - 5th-year Computer Science and Network Engineering student

**License**: MIT License

**Built with ❤️ for telecom fraud detection**