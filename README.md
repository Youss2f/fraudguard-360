# FraudGuard-360: Production-Ready Fraud Detection Platform

[![CI Pipeline](https://github.com/Youss2f/fraudguard-360/actions/workflows/ci.yml/badge.svg)](https://github.com/Youss2f/fraudguard-360/actions/workflows/ci.yml)
[![Code Quality](https://github.com/Youss2f/fraudguard-360/actions/workflows/quality.yml/badge.svg)](https://github.com/Youss2f/fraudguard-360/actions/workflows/quality.yml)
[![Release](https://github.com/Youss2f/fraudguard-360/actions/workflows/deploy.yml/badge.svg)](https://github.com/Youss2f/fraudguard-360/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Deployed-326CE5?logo=kubernetes&logoColor=white)](https://kubernetes.io/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-2.0+-231F20?logo=apache-kafka&logoColor=white)](https://kafka.apache.org/)

A cloud-native, real-time fraud detection platform leveraging microservices architecture, machine learning, and graph analytics. This enterprise-grade system processes transactions through a sophisticated pipeline to identify fraudulent activities with sub-second latency.

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Architecture

### System Architecture Diagram

```
┌─────────────┐         ┌──────────────────┐         ┌──────────────────┐
│   Client    │────────▶│   API Gateway    │────────▶│      Kafka       │
│ Application │         │   (FastAPI)      │         │ (Event Streaming)│
└─────────────┘         └──────────────────┘         └──────────────────┘
                                                               │
                                                               ▼
                        ┌──────────────────────────────────────────┐
                        │      Risk Scoring Service                │
                        │  (Rule-based Fraud Detection)            │
                        └──────────────────────────────────────────┘
                                         │
                                         ▼
                        ┌──────────────────────────────────────────┐
                        │          Kafka Output Topic              │
                        │     (Scored Transactions)                │
                        └──────────────────────────────────────────┘
                                         │
                      ┌──────────────────┴──────────────────┐
                      ▼                                     ▼
         ┌────────────────────┐              ┌────────────────────┐
         │   ML Service       │              │    PostgreSQL      │
         │ (Graph Analytics)  │              │   (Transaction     │
         │                    │              │    Storage)        │
         └────────────────────┘              └────────────────────┘
                  │
                  ▼
         ┌────────────────────┐
         │      Neo4j         │
         │  (Graph Database)  │
         └────────────────────┘
```

### Architecture Highlights

- **Microservices Architecture**: Independent, scalable services
- **Event-Driven Design**: Asynchronous processing via Kafka
- **Polyglot Persistence**: PostgreSQL for transactions, Neo4j for graph analytics, Redis for caching
- **Cloud-Native**: Containerized with Docker, orchestrated with Kubernetes
- **Observability**: Prometheus metrics, structured logging

## Features

### Core Capabilities

- **Real-time Transaction Processing**: Sub-second fraud detection
- **Rule-based Scoring**: Configurable fraud detection rules
- **Asynchronous Pipeline**: Kafka-based event streaming
- **Horizontal Scalability**: Auto-scaling based on load
- **Health Monitoring**: Comprehensive health checks and metrics
- **API-First Design**: RESTful API with OpenAPI documentation

### DevOps & Infrastructure

- **CI/CD Pipeline**: Automated testing, building, and deployment
- **Container Registry**: GitHub Container Registry integration
- **Infrastructure as Code**: Terraform for OCI provisioning
- **Kubernetes Deployment**: Production-ready manifests
- **Helm Charts**: Parameterized deployments
- **Security Scanning**: Trivy and CodeQL integration

## Quick Start

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- Git
- 8GB RAM minimum
- 20GB disk space

### Local Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/Youss2f/fraudguard-360.git
cd fraudguard-360
```

2. **Start the complete infrastructure**

```bash
docker-compose up -d
```

This will start:
- Kafka & Zookeeper
- PostgreSQL database
- Redis cache
- Neo4j graph database
- API Gateway (port 8000)
- Risk Scoring Service
- ML Service
- Prometheus (port 9090)

3. **Verify all services are running**

```bash
docker-compose ps
```

4. **Test the API**

```bash
# Health check
curl http://localhost:8000/health

# Submit a test transaction
curl -X POST http://localhost:8000/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "USR_12345",
    "amount": 250.50,
    "location": "New York, NY",
    "merchant_id": "MERCH_789"
  }'
```

5. **View metrics**

```bash
# API Gateway metrics
curl http://localhost:8000/metrics

# Prometheus UI
open http://localhost:9090
```

### Stopping the Environment

```bash
docker-compose down
```

To also remove volumes:

```bash
docker-compose down -v
```

## Deployment

### Kubernetes Deployment

#### Option 1: Using kubectl (Direct Kubernetes Manifests)

```bash
# Apply the production deployment
kubectl apply -f infrastructure/kubernetes/production-deployment.yaml

# Check deployment status
kubectl get pods -n fraudguard
kubectl get services -n fraudguard

# Access the API Gateway
kubectl port-forward -n fraudguard svc/api-gateway 8000:80
```

#### Option 2: Using Helm (Recommended)

```bash
# Install with default values
helm install fraudguard ./helm/fraudguard

# Install with custom values
helm install fraudguard ./helm/fraudguard \
  --set apiGateway.replicaCount=5 \
  --set global.imageTag=v1.0.0

# Upgrade deployment
helm upgrade fraudguard ./helm/fraudguard

# Uninstall
helm uninstall fraudguard
```

#### Helm Configuration

Create a custom `values.yaml`:

```yaml
apiGateway:
  replicaCount: 3
  image:
    tag: latest
  service:
    type: LoadBalancer

riskScoringService:
  replicaCount: 2
  autoscaling:
    enabled: true
    maxReplicas: 10
```

Then deploy:

```bash
helm install fraudguard ./helm/fraudguard -f values.yaml
```

### Cloud Deployment (Oracle Cloud Infrastructure)

#### Provision Infrastructure with Terraform

```bash
cd terraform

# Initialize Terraform
terraform init

# Create terraform.tfvars with your OCI credentials
cat > terraform.tfvars <<EOF
tenancy_ocid     = "ocid1.tenancy.oc1..aaaaaaaa..."
user_ocid        = "ocid1.user.oc1..aaaaaaaa..."
fingerprint      = "xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx"
private_key_path = "~/.oci/oci_api_key.pem"
region           = "us-ashburn-1"
compartment_ocid = "ocid1.compartment.oc1..aaaaaaaa..."
EOF

# Plan the infrastructure
terraform plan

# Apply the configuration
terraform apply

# Get outputs
terraform output
```

This will create:
- Virtual Cloud Network (VCN)
- Public and private subnets
- Internet gateway
- Security lists configured for Kubernetes

#### Deploy to OKE (Oracle Kubernetes Engine)

After infrastructure is provisioned:

```bash
# Configure kubectl with OKE cluster
oci ce cluster create-kubeconfig \
  --cluster-id <your-cluster-ocid> \
  --file ~/.kube/config \
  --region us-ashburn-1

# Deploy using Helm
helm install fraudguard ./helm/fraudguard \
  --set global.imageRegistry=ghcr.io \
  --set global.imageOwner=youss2f
```

## API Documentation

Comprehensive API documentation is available at [`docs/api/API_DOCUMENTATION.md`](docs/api/API_DOCUMENTATION.md).

### Quick API Reference

#### Submit Transaction

```bash
POST /transactions
Content-Type: application/json

{
  "user_id": "USR_12345",
  "amount": 250.50,
  "location": "New York, NY",
  "merchant_id": "MERCH_789",
  "transaction_type": "purchase"
}
```

#### Response

```json
{
  "transaction_id": "TXN_1698840000000_USR_12345",
  "status": "accepted",
  "message": "Transaction submitted for fraud analysis",
  "timestamp": "2025-11-01T12:00:00.000Z"
}
```

See full API documentation for:
- All endpoints
- Request/response schemas
- Error codes
- Rate limiting
- Integration examples (Python, JavaScript, cURL)

## Development

### Running Tests

```bash
# Run all tests with coverage
docker-compose exec api-gateway pytest tests/ -v --cov=. --cov-report=term-missing

# Run specific test file
docker-compose exec api-gateway pytest tests/test_integration.py -v

# Run tests for all services
./scripts/run-all-tests.sh
```

### Code Quality

```bash
# Linting
flake8 src/

# Security scanning
bandit -r src/

# Type checking
mypy src/
```

### Adding New Services

1. Create service directory: `src/new-service/`
2. Add `Dockerfile`, `requirements.txt`, and source code
3. Update `docker-compose.yml`
4. Add Kubernetes manifests
5. Update Helm chart
6. Add to CI/CD pipeline

### Local Development Workflow

```bash
# Make changes to code
vim src/api-gateway/app.py

# Rebuild specific service
docker-compose up -d --build api-gateway

# View logs
docker-compose logs -f api-gateway

# Run tests
docker-compose exec api-gateway pytest tests/ -v
```

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.9+ |
| **Frameworks** | FastAPI, Uvicorn |
| **Message Broker** | Apache Kafka, Zookeeper |
| **Databases** | PostgreSQL, Redis, Neo4j |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes, Helm |
| **Infrastructure as Code** | Terraform (OCI Provider) |
| **CI/CD** | GitHub Actions |
| **Monitoring** | Prometheus, Grafana (optional) |
| **Security** | Trivy, CodeQL, Bandit |
| **Testing** | Pytest, pytest-asyncio, pytest-cov |
| **Code Quality** | Flake8, Black, MyPy |

## Project Structure

```
fraudguard-360/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline
├── docs/
│   ├── api/
│   │   └── API_DOCUMENTATION.md      # Complete API docs
│   └── architecture/
│       └── system-architecture.md    # Architecture details
├── helm/
│   └── fraudguard/
│       ├── Chart.yaml                # Helm chart metadata
│       ├── values.yaml               # Default configuration
│       └── templates/                # Kubernetes templates
├── infrastructure/
│   ├── kubernetes/
│   │   └── production-deployment.yaml
│   └── prometheus/
│       └── prometheus.yml
├── src/
│   ├── api-gateway/
│   │   ├── app.py                    # Main application
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── tests/
│   │       ├── test_utils.py         # Unit tests
│   │       └── test_integration.py   # Integration tests
│   ├── risk-scoring-service/
│   │   ├── app.py                    # Kafka consumer + scoring
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── ml-service/
│       ├── Dockerfile
│       └── requirements.txt
├── terraform/
│   ├── main.tf                       # OCI infrastructure
│   ├── variables.tf
│   ├── outputs.tf
│   └── README.md
├── docker-compose.yml                # Local development
└── README.md                         # This file
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'feat: Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Monitoring & Observability

### Prometheus Metrics

Access metrics at:
- API Gateway: `http://localhost:8000/metrics`
- Risk Scoring: `http://localhost:8001/metrics`
- Prometheus UI: `http://localhost:9090`

### Key Metrics

- `api_requests_total`: Total API requests by endpoint and status
- `transactions_total`: Total transactions processed
- `fraud_score`: Distribution of fraud scores
- `request_duration_seconds`: Request latency histogram
- `high_risk_transactions_total`: Count of high-risk transactions

### Grafana Dashboards (Optional)

Import pre-built dashboards for comprehensive monitoring (coming soon).

## Security

### Security Features

- Regular vulnerability scanning (Trivy)
- Static code analysis (CodeQL, Bandit)
- Dependency updates via Dependabot
- Secret management with Kubernetes secrets
- Network policies for service isolation
- TLS/SSL for external communications

### Reporting Security Issues

Please report security vulnerabilities to: security@fraudguard.io

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with best practices from the DevOps & Cloud Engineering community
- Inspired by real-world fraud detection systems
- Powered by open-source technologies

## Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/Youss2f/fraudguard-360/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Youss2f/fraudguard-360/discussions)

---

**Built for the DevOps community**
