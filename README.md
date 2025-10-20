# FraudGuard 360: A Cloud-Native, Real-Time Fraud Detection Platform

[CI & Security Scan Status](https://github.com/Youss2f/fraudguard-360/actions/workflows/ci.yml)

This repository contains the source code and infrastructure for FraudGuard 360, a real-time, microservices-based platform designed to detect fraudulent activities in telecom networks. The project serves as a modern, scalable replacement for a legacy batch-processing system, leveraging a streaming architecture to provide sub-second detection latency.

## Architecture Overview

The system is built on a decoupled, microservices architecture orchestrated by Kubernetes. Data flows asynchronously through Apache Kafka, is processed in real-time by stream processors, and analyzed for complex fraud networks using a Neo4j graph database and a GraphSAGE-based Machine Learning model.

A detailed architecture diagram can be found in /docs/architecture/system-architecture.md.

## Local Development Quickstart

### Prerequisites
- Docker
- Docker Compose

### Instructions
1.  **Clone the repository:**
    `sh
    git clone https://github.com/Youss2f/fraudguard-360.git
    cd fraudguard-360
    `
2.  **Launch the platform:**
    `sh
    docker-compose up --build
    `
3.  **Access Services:**
    -   API Gateway: http://localhost:8000
    -   Neo4j Browser: http://localhost:7474

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | Docker, Kubernetes | Containerization and deployment |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **Messaging** | Apache Kafka | Real-time data streaming |
| **Database** | Neo4j | Graph-based fraud network analysis |
| **Backend** | Python, FastAPI | Microservices implementation |
| **Security** | Trivy, CodeQL | Vulnerability scanning and code analysis |

## Development

### Running Tests
`ash
# Run tests for all services
docker-compose exec api-gateway pytest tests/
docker-compose exec ml-service pytest tests/
docker-compose exec risk-scoring-service pytest tests/
`

### Code Quality
`ash
flake8 src/
bandit -r src/
`

## Deployment

The platform supports multiple deployment scenarios:

### Local Development
`ash
docker-compose up --build
`

### Production (Kubernetes)
`ash
kubectl apply -f kubernetes/
`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.
