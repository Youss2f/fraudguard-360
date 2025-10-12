---
layout: default
title: Deployment
nav_order: 5
description: "Production deployment guide for FraudGuard 360°"
---

# Production Deployment
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Deployment Overview

FraudGuard 360° supports multiple deployment strategies for production environments, from single-server deployments to highly available Kubernetes clusters.

## Docker Compose Deployment
{: .d-inline-block }

Simple Production
{: .label .label-green }

Suitable for small to medium-scale deployments.

### Prerequisites

- Docker 24.0+ and Docker Compose 2.20+
- Minimum 8GB RAM, 4 CPU cores
- 50GB+ storage space
- SSL certificates for HTTPS

### 1. Production Configuration

Create production environment file:

```bash
# Create production environment
cp .env.example .env.production

# Edit production settings
nano .env.production
```

**Production Environment Variables:**
```bash
# Environment
NODE_ENV=production
ENVIRONMENT=production

# Security
JWT_SECRET_KEY=your-very-secure-secret-key-here
API_RATE_LIMIT=5000
CORS_ORIGINS=https://your-domain.com

# Database URLs (Production)
POSTGRES_URL=postgresql://fraud_user:secure_password@postgres:5432/fraudguard_prod
REDIS_URL=redis://:secure_redis_password@redis:6379/0
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=secure_neo4j_password

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=PLAIN

# SSL/TLS
SSL_CERT_PATH=/etc/ssl/certs/fraudguard.crt
SSL_KEY_PATH=/etc/ssl/private/fraudguard.key

# Monitoring
PROMETHEUS_RETENTION=30d
GRAFANA_ADMIN_PASSWORD=secure_grafana_password

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### 2. Production Docker Compose

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # Production API Gateway with SSL
  api-gateway:
    build:
      context: ./api-gateway
      dockerfile: Dockerfile.prod
    ports:
      - "443:443"
      - "80:80"
    environment:
      - SSL_CERT_PATH=/etc/ssl/certs/fraudguard.crt
      - SSL_KEY_PATH=/etc/ssl/private/fraudguard.key
    volumes:
      - ./ssl:/etc/ssl
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "https://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ML Inference Service
  ml-service:
    build:
      context: ./core-ml-service
      dockerfile: Dockerfile.prod
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    restart: unless-stopped

  # Risk Scoring Service
  risk-service:
    build:
      context: ./risk-scoring-service
      dockerfile: Dockerfile.prod
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    restart: unless-stopped

  # Graph Analytics Service
  graph-service:
    build:
      context: ./graph-analytics-service
      dockerfile: Dockerfile.prod
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
    restart: unless-stopped

  # Production PostgreSQL
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: fraudguard_prod
      POSTGRES_USER: fraud_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./postgres/init-scripts:/docker-entrypoint-initdb.d
    restart: unless-stopped
    command: >
      postgres
      -c max_connections=100
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c wal_level=replica
      -c max_wal_senders=3

  # Production Redis
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD} --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Production Neo4j
  neo4j:
    image: neo4j:5.15-enterprise
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_dbms_memory_heap_initial__size: 1G
      NEO4J_dbms_memory_heap_max__size: 2G
      NEO4J_dbms_memory_pagecache_size: 1G
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

  # Production Kafka
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_NUM_PARTITIONS: 8
    volumes:
      - kafka_data:/var/lib/kafka/data
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
  kafka_data:
  grafana_data:
  prometheus_data:

networks:
  fraudguard-prod:
    driver: bridge
```

### 3. SSL Certificate Setup

```bash
# Generate self-signed certificate (for testing)
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/fraudguard.key \
  -out ssl/fraudguard.crt \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=your-domain.com"

# Or use Let's Encrypt (recommended)
certbot certonly --standalone -d your-domain.com
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/fraudguard.crt
cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/fraudguard.key
```

### 4. Deploy to Production

```bash
# Build and start production services
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps

# Check logs
docker-compose -f docker-compose.production.yml logs -f
```

## Kubernetes Deployment
{: .d-inline-block }

Enterprise Scale
{: .label .label-blue }

For high-availability and auto-scaling production environments.

### Prerequisites

- Kubernetes 1.25+
- kubectl configured
- Helm 3.0+
- Ingress controller (NGINX, Traefik)
- Storage class for persistent volumes

### 1. Namespace and Secrets

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: fraudguard-prod
  labels:
    name: fraudguard-prod
---
# Secrets
apiVersion: v1
kind: Secret
metadata:
  name: fraudguard-secrets
  namespace: fraudguard-prod
type: Opaque
data:
  jwt-secret: <base64-encoded-secret>
  postgres-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  neo4j-password: <base64-encoded-password>
```

### 2. ConfigMaps

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fraudguard-config
  namespace: fraudguard-prod
data:
  API_RATE_LIMIT: "5000"
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  POSTGRES_DB: "fraudguard_prod"
  REDIS_DB: "0"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
```

### 3. Database Deployments

```yaml
# k8s/postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: fraudguard-prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: fraudguard-config
              key: POSTGRES_DB
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: fraudguard-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: fraudguard-prod
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

### 4. Application Deployments

```yaml
# k8s/ml-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: fraudguard-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: ml-service
        image: fraudguard360/ml-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: ENVIRONMENT
          value: "production"
        envFrom:
        - configMapRef:
            name: fraudguard-config
        - secretRef:
            name: fraudguard-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: fraudguard-prod
spec:
  selector:
    app: ml-service
  ports:
  - protocol: TCP
    port: 8001
    targetPort: 8001
  type: ClusterIP
```

### 5. Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-service-hpa
  namespace: fraudguard-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 6. Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fraudguard-ingress
  namespace: fraudguard-prod
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.fraudguard360.com
    secretName: fraudguard-tls
  rules:
  - host: api.fraudguard360.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-gateway
            port:
              number: 8000
      - path: /ml
        pathType: Prefix
        backend:
          service:
            name: ml-service
            port:
              number: 8001
```

### 7. Deploy to Kubernetes

```bash
# Create namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml

# Deploy databases
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/neo4j-deployment.yaml

# Deploy application services
kubectl apply -f k8s/ml-service-deployment.yaml
kubectl apply -f k8s/risk-service-deployment.yaml
kubectl apply -f k8s/graph-service-deployment.yaml
kubectl apply -f k8s/api-gateway-deployment.yaml

# Deploy autoscaling and ingress
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# Verify deployment
kubectl get pods -n fraudguard-prod
kubectl get services -n fraudguard-prod
kubectl get ingress -n fraudguard-prod
```

## Helm Chart Deployment
{: .d-inline-block }

Simplified K8s
{: .label .label-purple }

Using Helm for simplified Kubernetes deployment.

### 1. Install with Helm

```bash
# Add Helm repository
helm repo add fraudguard360 https://charts.fraudguard360.com
helm repo update

# Create values file
cat > values-production.yaml << EOF
replicaCount:
  mlService: 3
  riskService: 2
  graphService: 2
  apiGateway: 2

image:
  repository: fraudguard360
  tag: "1.2.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: api.fraudguard360.com
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls:
    - secretName: fraudguard-tls
      hosts:
        - api.fraudguard360.com

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

resources:
  mlService:
    limits:
      cpu: 1
      memory: 2Gi
    requests:
      cpu: 500m
      memory: 1Gi

persistence:
  enabled: true
  storageClass: gp2
  postgres:
    size: 100Gi
  neo4j:
    size: 50Gi
  redis:
    size: 10Gi

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: secure_password
EOF

# Install FraudGuard 360°
helm install fraudguard360 fraudguard360/fraudguard \
  -f values-production.yaml \
  --namespace fraudguard-prod \
  --create-namespace

# Check deployment status
helm status fraudguard360 -n fraudguard-prod
```

## Cloud Provider Deployments

### AWS EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster \
  --name fraudguard360-prod \
  --version 1.25 \
  --region us-west-2 \
  --nodegroup-name workers \
  --node-type m5.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name fraudguard360-prod

# Install AWS Load Balancer Controller
kubectl apply -f https://github.com/kubernetes-sigs/aws-load-balancer-controller/releases/download/v2.4.4/v2_4_4_full.yaml

# Deploy application
kubectl apply -f k8s/aws/
```

### Azure AKS Deployment

```bash
# Create resource group
az group create --name fraudguard360-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group fraudguard360-rg \
  --name fraudguard360-aks \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group fraudguard360-rg --name fraudguard360-aks

# Deploy application
kubectl apply -f k8s/azure/
```

### Google GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create fraudguard360-cluster \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --zone us-central1-a \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials fraudguard360-cluster --zone us-central1-a

# Deploy application
kubectl apply -f k8s/gcp/
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "fraud_detection_rules.yml"
  - "system_alerts.yml"

scrape_configs:
  - job_name: 'fraudguard-api-gateway'
    static_configs:
      - targets: ['api-gateway:8000']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'fraudguard-ml-service'
    static_configs:
      - targets: ['ml-service:8001']
    metrics_path: /metrics

  - job_name: 'fraudguard-postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "FraudGuard 360° Production Dashboard",
    "panels": [
      {
        "title": "Fraud Detection Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(fraud_detections_total[5m])",
            "legendFormat": "Frauds/sec"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## Security Hardening

### 1. Network Security

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: fraudguard-network-policy
  namespace: fraudguard-prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### 2. Pod Security Standards

```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: fraudguard-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

### 3. Secrets Management

```bash
# Use external secret management
# Example with AWS Secrets Manager
kubectl create secret generic fraudguard-secrets \
  --from-literal=jwt-secret="$(aws secretsmanager get-secret-value --secret-id prod/fraudguard/jwt-secret --query SecretString --output text)" \
  --from-literal=db-password="$(aws secretsmanager get-secret-value --secret-id prod/fraudguard/db-password --query SecretString --output text)"
```

## Backup and Disaster Recovery

### Database Backups

```bash
# PostgreSQL backup script
#!/bin/bash
BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
kubectl exec -n fraudguard-prod postgres-0 -- pg_dump -U fraud_user fraudguard_prod > $BACKUP_DIR/fraudguard_$DATE.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/fraudguard_$DATE.sql s3://fraudguard-backups/postgres/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
```

### Neo4j Backup

```bash
# Neo4j backup script
#!/bin/bash
BACKUP_DIR="/backups/neo4j"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
kubectl exec -n fraudguard-prod neo4j-0 -- neo4j-admin backup --backup-dir=/backups --name=graph_$DATE

# Upload to S3
aws s3 sync $BACKUP_DIR s3://fraudguard-backups/neo4j/
```

## Performance Optimization

### 1. Resource Allocation

```yaml
# Optimized resource allocation
resources:
  mlService:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"
  
  postgres:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "1"
      memory: "2Gi"
```

### 2. Caching Strategy

```yaml
# Redis cluster for high availability
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
spec:
  replicas: 6
  serviceName: redis-cluster
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - redis-server
        - /etc/redis/redis.conf
        - --cluster-enabled
        - "yes"
```

### 3. Database Optimization

```sql
-- PostgreSQL optimization
-- In postgresql.conf
shared_buffers = 1GB
effective_cache_size = 3GB
maintenance_work_mem = 256MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

## Health Checks and Monitoring

### Application Health Checks

```bash
#!/bin/bash
# Health check script
echo "=== FraudGuard 360° Production Health Check ==="

# Check service endpoints
services=("api-gateway:8000" "ml-service:8001" "risk-service:8002" "graph-service:8003")

for service in "${services[@]}"; do
    endpoint="https://$service/health"
    response=$(curl -s -o /dev/null -w "%{http_code}" $endpoint)
    
    if [ $response -eq 200 ]; then
        echo "✅ $service: Healthy"
    else
        echo "❌ $service: Unhealthy (HTTP $response)"
    fi
done

# Check database connections
echo "🔍 Database connectivity:"
kubectl exec -n fraudguard-prod postgres-0 -- pg_isready -U fraud_user
kubectl exec -n fraudguard-prod redis-0 -- redis-cli ping
kubectl exec -n fraudguard-prod neo4j-0 -- cypher-shell "RETURN 1"
```

### Alerting Rules

```yaml
# prometheus-alerts.yml
groups:
- name: fraudguard.rules
  rules:
  - alert: HighFraudDetectionRate
    expr: rate(fraud_detections_total[5m]) > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High fraud detection rate detected"
      
  - alert: APIHighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "API latency is above 1 second"
      
  - alert: DatabaseConnections
    expr: pg_stat_activity_count > 80
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "PostgreSQL connection count is high"
```

---

This comprehensive deployment guide covers production-ready deployments from simple Docker Compose setups to enterprise Kubernetes clusters with monitoring, security, and disaster recovery capabilities.