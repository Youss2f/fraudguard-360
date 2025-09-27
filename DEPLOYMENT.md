# FraudGuard 360 - Deployment Guide

## Overview

This guide covers deploying the FraudGuard 360 fraud detection platform using Kubernetes and Helm. The platform supports multiple deployment environments with progressive deployment strategies.

## Architecture

### Microservices Components
- **API Gateway**: Central authentication and routing (FastAPI)
- **Frontend**: React TypeScript dashboard with real-time analytics
- **ML Service**: Machine learning inference service with GraphSAGE
- **Flink Jobs**: Real-time stream processing for fraud detection
- **PostgreSQL**: Relational data storage
- **Neo4j**: Graph database for relationship analysis
- **Apache Kafka**: Message streaming and event processing

### Infrastructure
- **Kubernetes**: Container orchestration (v1.18+)
- **Helm**: Package management and templating (v3.13+)
- **NGINX Ingress**: External access and SSL termination
- **Prometheus & Grafana**: Monitoring and observability

## Prerequisites

### Kubernetes Cluster
```bash
# Minimum requirements:
# - Kubernetes 1.18+
# - 16 vCPUs, 32GB RAM for production
# - 100GB+ storage with dynamic provisioning
# - LoadBalancer support for ingress

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### Required Tools
```bash
# Install Helm
curl https://get.helm.sh/helm-v3.13.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/

# Install kubectl (if not already installed)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

### Container Registry
```bash
# Example with GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build and push images
docker build -t ghcr.io/your-org/fraudguard-360/api-gateway:v1.0.0 ./api-gateway
docker build -t ghcr.io/your-org/fraudguard-360/frontend:v1.0.0 ./frontend
docker build -t ghcr.io/your-org/fraudguard-360/ml-service:v1.0.0 ./ml-service

docker push ghcr.io/your-org/fraudguard-360/api-gateway:v1.0.0
docker push ghcr.io/your-org/fraudguard-360/frontend:v1.0.0
docker push ghcr.io/your-org/fraudguard-360/ml-service:v1.0.0
```

## Quick Start

### 1. Prepare Environment
```bash
# Clone repository
git clone https://github.com/your-org/fraudguard-360.git
cd fraudguard-360

# Add Helm repositories for dependencies
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add neo4j https://helm.neo4j.com/neo4j
helm repo update
```

### 2. Configure Values
```bash
# Copy and customize values for your environment
cp helm-chart/values.yaml helm-chart/values-custom.yaml

# Edit values-custom.yaml:
# - Set image registry and tags
# - Configure ingress hostnames
# - Adjust resource limits
# - Set database passwords
```

### 3. Deploy to Development
```bash
# Install dependencies
helm dependency update ./helm-chart

# Deploy development environment
helm install fraudguard-dev ./helm-chart \
  --namespace fraudguard-dev \
  --create-namespace \
  --values ./helm-chart/values.yaml \
  --values ./helm-chart/values-dev.yaml \
  --set global.imageRegistry=ghcr.io/your-org/fraudguard-360/ \
  --wait

# Verify deployment
kubectl get pods -n fraudguard-dev
kubectl get services -n fraudguard-dev
```

### 4. Access Services
```bash
# Port forward for local access
kubectl port-forward -n fraudguard-dev svc/fraudguard-dev-frontend 3000:3000 &
kubectl port-forward -n fraudguard-dev svc/fraudguard-dev-api-gateway 8000:8000 &

# Access dashboard: http://localhost:3000
# Access API: http://localhost:8000/docs
```

## Production Deployment

### 1. Infrastructure Prerequisites
```bash
# Install NGINX Ingress Controller
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.replicaCount=3 \
  --set controller.nodeSelector."kubernetes\.io/os"=linux

# Install cert-manager for SSL
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true

# Create cluster issuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 2. Secrets Management
```bash
# Create database secrets
kubectl create secret generic fraudguard-postgresql-secret \
  --from-literal=password=YOUR_SECURE_DB_PASSWORD \
  --namespace fraudguard-production

kubectl create secret generic fraudguard-neo4j-secret \
  --from-literal=username=neo4j \
  --from-literal=password=YOUR_SECURE_NEO4J_PASSWORD \
  --namespace fraudguard-production

# Create admin auth secret for admin ingress
htpasswd -c auth admin
kubectl create secret generic fraudguard-admin-auth \
  --from-file=auth \
  --namespace fraudguard-production
```

### 3. Production Deployment
```bash
# Deploy to production
helm install fraudguard-prod ./helm-chart \
  --namespace fraudguard-production \
  --create-namespace \
  --values ./helm-chart/values.yaml \
  --values ./helm-chart/values-production.yaml \
  --set global.imageRegistry=ghcr.io/your-org/fraudguard-360/ \
  --set api-gateway.image.tag=v1.0.0 \
  --set frontend.image.tag=v1.0.0 \
  --set ml-service.image.tag=v1.0.0 \
  --timeout=15m \
  --wait

# Verify deployment
kubectl get pods -n fraudguard-production
kubectl get ingress -n fraudguard-production
```

### 4. Post-Deployment Verification
```bash
# Check all pods are running
kubectl wait --for=condition=ready pod \
  -l app.kubernetes.io/part-of=fraudguard-360 \
  -n fraudguard-production \
  --timeout=300s

# Test API endpoints
curl -f https://api.fraudguard.yourdomain.com/health
curl -f https://fraudguard.yourdomain.com

# Check logs
kubectl logs -n fraudguard-production -l app.kubernetes.io/name=api-gateway --tail=100
kubectl logs -n fraudguard-production -l app.kubernetes.io/name=ml-service --tail=100
```

## Environment-Specific Configurations

### Development
- Single replica for all services
- Minimal resource allocation
- No ingress (use port-forwarding)
- Simplified database setup
- No network policies

### Staging  
- 2-3 replicas for critical services
- Moderate resource allocation
- Ingress with staging hostnames
- Full feature testing
- Network policies enabled

### Production
- High availability (3+ replicas)
- Optimized resource allocation
- Production ingress with SSL
- Full monitoring and backup
- Complete security hardening

## Monitoring and Observability

### Prometheus Metrics
```bash
# Install Prometheus and Grafana
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Import FraudGuard dashboards
kubectl apply -f monitoring/grafana-dashboards/
```

### Log Aggregation
```bash
# Install ELK stack or equivalent
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --create-namespace

helm install kibana elastic/kibana \
  --namespace logging
```

## Scaling and Optimization

### Horizontal Pod Autoscaling
```bash
# HPA is automatically configured in Helm templates
# Monitor scaling decisions
kubectl get hpa -n fraudguard-production
kubectl describe hpa fraudguard-prod-api-gateway -n fraudguard-production
```

### Resource Optimization
```bash
# Monitor resource usage
kubectl top pods -n fraudguard-production
kubectl top nodes

# Adjust resource requests and limits in values files
# Redeploy with updated configuration
helm upgrade fraudguard-prod ./helm-chart \
  --values ./helm-chart/values-production.yaml
```

## Backup and Disaster Recovery

### Database Backups
```bash
# PostgreSQL backup
kubectl exec -n fraudguard-production fraudguard-prod-postgresql-0 -- \
  pg_dump -U fraudguard fraudguard > backup-$(date +%Y%m%d).sql

# Neo4j backup
kubectl exec -n fraudguard-production fraudguard-prod-neo4j-0 -- \
  neo4j-admin backup --backup-dir=/backups --name=graph-$(date +%Y%m%d)
```

### Application State
```bash
# Backup Kafka topics and consumer offsets
kubectl exec -n fraudguard-production fraudguard-prod-kafka-0 -- \
  kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic fraud-transactions --from-beginning > kafka-backup.json
```

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod status and events
kubectl describe pod POD_NAME -n fraudguard-production
kubectl logs POD_NAME -n fraudguard-production --previous

# Common causes:
# - Image pull errors (check registry access)
# - Resource limits too low
# - Configuration errors
# - Database connectivity issues
```

#### Service Communication
```bash
# Test service connectivity
kubectl exec -n fraudguard-production FRONTEND_POD -- \
  curl -f http://fraudguard-prod-api-gateway:8000/health

# Check network policies
kubectl get networkpolicy -n fraudguard-production
kubectl describe networkpolicy api-gateway-network-policy -n fraudguard-production
```

#### Performance Issues
```bash
# Check resource utilization
kubectl top pods -n fraudguard-production
kubectl get hpa -n fraudguard-production

# Analyze metrics in Grafana
# Scale services if needed
kubectl scale deployment fraudguard-prod-api-gateway --replicas=10 -n fraudguard-production
```

### Emergency Procedures

#### Rolling Back Deployment
```bash
# View deployment history
helm history fraudguard-prod -n fraudguard-production

# Rollback to previous version
helm rollback fraudguard-prod 1 -n fraudguard-production
```

#### Database Recovery
```bash
# Restore PostgreSQL from backup
kubectl exec -i fraudguard-prod-postgresql-0 -n fraudguard-production -- \
  psql -U fraudguard fraudguard < backup-20240101.sql

# Restore Neo4j from backup
kubectl exec fraudguard-prod-neo4j-0 -n fraudguard-production -- \
  neo4j-admin restore --from=/backups/graph-20240101
```

## Security Considerations

### Network Security
- Network policies isolate service communication
- Ingress controller provides SSL termination
- Admin services require authentication and IP whitelisting

### Container Security
- Non-root containers with read-only filesystems
- Security contexts drop all capabilities
- Regular vulnerability scanning with Trivy

### Secrets Management
- Kubernetes secrets for sensitive data
- Consider external secret management (HashiCorp Vault, AWS Secrets Manager)
- Regular credential rotation

## CI/CD Integration

The included GitHub Actions workflow provides:
- Automated testing and code quality checks
- Container image building and security scanning
- Progressive deployment (dev → staging → production)
- Integration and performance testing
- Rollback capabilities

Configure the following secrets in your GitHub repository:
- `GITHUB_TOKEN`: For container registry access
- `KUBE_CONFIG_STAGING`: Base64-encoded kubeconfig for staging
- `KUBE_CONFIG_PRODUCTION`: Base64-encoded kubeconfig for production
- `SLACK_WEBHOOK_URL`: For deployment notifications

## Support and Maintenance

### Regular Maintenance Tasks
1. **Weekly**: Review monitoring dashboards and alerts
2. **Monthly**: Update container images and Helm charts
3. **Quarterly**: Review and rotate secrets
4. **Annually**: Kubernetes cluster upgrades

### Health Checks
```bash
# Daily health check script
#!/bin/bash
kubectl get pods -n fraudguard-production | grep -v Running
kubectl get pvc -n fraudguard-production | grep -v Bound
kubectl top nodes
kubectl top pods -n fraudguard-production
```

For additional support, consult:
- Application logs via `kubectl logs`
- Monitoring dashboards in Grafana
- Project documentation and README files
- GitHub Issues for bug reports and feature requests