#!/bin/bash

# FraudGuard 360 - Complete Deployment Automation Script
# This script deploys the entire fraud detection platform

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="fraudguard-360"
NAMESPACE="fraud-detection"
KAFKA_REPLICAS=3
REDIS_REPLICAS=1
NEO4J_REPLICAS=1
POSTGRES_REPLICAS=1

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for deployment
wait_for_deployment() {
    local deployment=$1
    local namespace=$2
    print_status "Waiting for deployment $deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $namespace
}

# Function to wait for statefulset
wait_for_statefulset() {
    local statefulset=$1
    local namespace=$2
    local replicas=$3
    print_status "Waiting for statefulset $statefulset to be ready..."
    kubectl wait --for=jsonpath='{.status.readyReplicas}'=$replicas statefulset/$statefulset -n $namespace --timeout=600s
}

# Pre-deployment checks
print_status "Starting FraudGuard 360 deployment..."
print_status "Performing pre-deployment checks..."

# Check required tools
REQUIRED_TOOLS=("docker" "kubectl" "helm" "mvn" "npm" "python3")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command_exists $tool; then
        print_error "$tool is not installed. Please install it and try again."
        exit 1
    fi
done

print_success "All required tools are available"

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    print_error "Docker daemon is not running. Please start Docker and try again."
    exit 1
fi

print_success "Docker daemon is running"

# Check Kubernetes cluster connectivity
if ! kubectl cluster-info >/dev/null 2>&1; then
    print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
    exit 1
fi

print_success "Kubernetes cluster is accessible"

# Create namespace
print_status "Creating namespace $NAMESPACE..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Add Helm repositories
print_status "Adding Helm repositories..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add elastic https://helm.elastic.co
helm repo update

print_success "Helm repositories added and updated"

# Deploy infrastructure components
print_status "Deploying infrastructure components..."

# Deploy PostgreSQL
print_status "Deploying PostgreSQL..."
helm upgrade --install postgres bitnami/postgresql \
    --namespace $NAMESPACE \
    --set auth.postgresPassword=fraudguard123 \
    --set auth.database=fraudguard \
    --set primary.persistence.size=20Gi \
    --set readReplicas.replicaCount=1 \
    --wait --timeout=10m

print_success "PostgreSQL deployed"

# Deploy Redis
print_status "Deploying Redis..."
helm upgrade --install redis bitnami/redis \
    --namespace $NAMESPACE \
    --set auth.password=fraudguard123 \
    --set master.persistence.size=8Gi \
    --set replica.replicaCount=$REDIS_REPLICAS \
    --wait --timeout=10m

print_success "Redis deployed"

# Deploy Neo4j
print_status "Deploying Neo4j..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j
  namespace: $NAMESPACE
spec:
  serviceName: neo4j
  replicas: $NEO4J_REPLICAS
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.11-community
        ports:
        - containerPort: 7474
        - containerPort: 7687
        env:
        - name: NEO4J_AUTH
          value: "neo4j/fraudguard123"
        - name: NEO4J_PLUGINS
          value: '["apoc", "graph-data-science"]'
        - name: NEO4J_dbms_security_procedures_unrestricted
          value: "apoc.*,gds.*"
        - name: NEO4J_dbms_memory_heap_initial__size
          value: "1G"
        - name: NEO4J_dbms_memory_heap_max__size
          value: "2G"
        volumeMounts:
        - name: neo4j-data
          mountPath: /data
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
  volumeClaimTemplates:
  - metadata:
      name: neo4j-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: neo4j
  namespace: $NAMESPACE
spec:
  ports:
  - port: 7474
    name: http
  - port: 7687
    name: bolt
  selector:
    app: neo4j
EOF

wait_for_statefulset "neo4j" $NAMESPACE $NEO4J_REPLICAS
print_success "Neo4j deployed"

# Deploy Kafka
print_status "Deploying Kafka..."
helm upgrade --install kafka bitnami/kafka \
    --namespace $NAMESPACE \
    --set replicaCount=$KAFKA_REPLICAS \
    --set persistence.size=10Gi \
    --set zookeeper.persistence.size=8Gi \
    --set kraft.enabled=false \
    --wait --timeout=15m

print_success "Kafka deployed"

# Deploy Monitoring Stack
print_status "Deploying monitoring stack..."

# Deploy Prometheus
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
    --namespace $NAMESPACE \
    --set prometheus.prometheusSpec.retention=30d \
    --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
    --set grafana.adminPassword=fraudguard123 \
    --wait --timeout=15m

print_success "Monitoring stack deployed"

# Build application images
print_status "Building application Docker images..."

# Build API Gateway
print_status "Building API Gateway..."
cd api-gateway
docker build -t fraudguard-api-gateway:latest .
cd ..

# Build ML Service
print_status "Building ML Service..."
cd ml-service
docker build -t fraudguard-ml-service:latest .
cd ..

# Build Flink Jobs
print_status "Building Flink Jobs..."
cd flink-jobs
mvn clean package -DskipTests
docker build -t fraudguard-flink-jobs:latest .
cd ..

# Build Frontend
print_status "Building Frontend..."
cd frontend
npm install
npm run build
docker build -t fraudguard-frontend:latest .
cd ..

print_success "All application images built"

# Deploy application components
print_status "Deploying application components..."

# Deploy API Gateway
print_status "Deploying API Gateway..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
    spec:
      containers:
      - name: api-gateway
        image: fraudguard-api-gateway:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          value: "postgresql://postgres:fraudguard123@postgres:5432/fraudguard"
        - name: REDIS_URL
          value: "redis://:fraudguard123@redis-master:6379"
        - name: NEO4J_URI
          value: "bolt://neo4j:7687"
        - name: NEO4J_USER
          value: "neo4j"
        - name: NEO4J_PASSWORD
          value: "fraudguard123"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: api-gateway
  namespace: $NAMESPACE
spec:
  selector:
    app: api-gateway
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
EOF

wait_for_deployment "api-gateway" $NAMESPACE
print_success "API Gateway deployed"

# Deploy ML Service
print_status "Deploying ML Service..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: $NAMESPACE
spec:
  replicas: 2
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
        image: fraudguard-ml-service:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 8005
        env:
        - name: DATABASE_URL
          value: "postgresql://postgres:fraudguard123@postgres:5432/fraudguard"
        - name: REDIS_URL
          value: "redis://:fraudguard123@redis-master:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8005
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8005
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: $NAMESPACE
spec:
  selector:
    app: ml-service
  ports:
  - port: 8005
    targetPort: 8005
  type: ClusterIP
EOF

wait_for_deployment "ml-service" $NAMESPACE
print_success "ML Service deployed"

# Deploy Flink Job Manager
print_status "Deploying Flink Job Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-jobmanager
  namespace: $NAMESPACE
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flink
      component: jobmanager
  template:
    metadata:
      labels:
        app: flink
        component: jobmanager
    spec:
      containers:
      - name: jobmanager
        image: flink:1.17.1-scala_2.12-java11
        ports:
        - containerPort: 6123
          name: rpc
        - containerPort: 6124
          name: blob-server
        - containerPort: 8081
          name: webui
        command: ["/opt/flink/bin/jobmanager.sh"]
        args: ["start-foreground"]
        env:
        - name: FLINK_PROPERTIES
          value: |
            jobmanager.rpc.address: flink-jobmanager
            taskmanager.numberOfTaskSlots: 4
            parallelism.default: 2
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: flink-jobmanager
  namespace: $NAMESPACE
spec:
  type: ClusterIP
  ports:
  - name: rpc
    port: 6123
  - name: blob-server
    port: 6124
  - name: webui
    port: 8081
  selector:
    app: flink
    component: jobmanager
EOF

wait_for_deployment "flink-jobmanager" $NAMESPACE

# Deploy Flink Task Manager
print_status "Deploying Flink Task Manager..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-taskmanager
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: flink
      component: taskmanager
  template:
    metadata:
      labels:
        app: flink
        component: taskmanager
    spec:
      containers:
      - name: taskmanager
        image: flink:1.17.1-scala_2.12-java11
        ports:
        - containerPort: 6122
          name: rpc
        - containerPort: 6125
          name: query-state
        command: ["/opt/flink/bin/taskmanager.sh"]
        args: ["start-foreground"]
        env:
        - name: FLINK_PROPERTIES
          value: |
            jobmanager.rpc.address: flink-jobmanager
            taskmanager.numberOfTaskSlots: 4
            parallelism.default: 2
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
EOF

wait_for_deployment "flink-taskmanager" $NAMESPACE
print_success "Flink cluster deployed"

# Deploy Frontend
print_status "Deploying Frontend..."
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: fraudguard-frontend:latest
        imagePullPolicy: Never
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: $NAMESPACE
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
EOF

wait_for_deployment "frontend" $NAMESPACE
print_success "Frontend deployed"

# Deploy Ingress
print_status "Deploying Ingress..."
cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fraudguard-ingress
  namespace: $NAMESPACE
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: fraudguard.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 80
      - path: /api
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
              number: 8005
      - path: /flink
        pathType: Prefix
        backend:
          service:
            name: flink-jobmanager
            port:
              number: 8081
EOF

print_success "Ingress deployed"

# Initialize database schema
print_status "Initializing database schema..."
kubectl exec -n $NAMESPACE deployment/postgres -- psql -U postgres -d fraudguard -c "
CREATE EXTENSION IF NOT EXISTS 'uuid-ossp';

-- CDR table
CREATE TABLE IF NOT EXISTS cdr_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    caller_id VARCHAR(50) NOT NULL,
    callee_id VARCHAR(50) NOT NULL,
    call_type VARCHAR(20) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    duration INTEGER NOT NULL,
    bytes_transmitted BIGINT,
    location_caller VARCHAR(100),
    location_callee VARCHAR(100),
    tower_id VARCHAR(50),
    device_imei VARCHAR(20),
    cost DECIMAL(10,2),
    country_code VARCHAR(5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fraud alerts table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(50) NOT NULL,
    fraud_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    risk_score DECIMAL(5,3) NOT NULL,
    confidence DECIMAL(5,3) NOT NULL,
    description TEXT,
    evidence JSONB,
    status VARCHAR(20) DEFAULT 'OPEN',
    call_id UUID,
    location VARCHAR(100),
    cost DECIMAL(10,2),
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    user_id VARCHAR(50) PRIMARY KEY,
    risk_level VARCHAR(20) DEFAULT 'LOW',
    fraud_probability DECIMAL(5,3) DEFAULT 0.0,
    total_calls INTEGER DEFAULT 0,
    total_alerts INTEGER DEFAULT 0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    profile_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cases table
CREATE TABLE IF NOT EXISTS fraud_cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_number VARCHAR(20) UNIQUE NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    priority VARCHAR(20) DEFAULT 'MEDIUM',
    status VARCHAR(20) DEFAULT 'OPEN',
    assignee_id VARCHAR(50),
    reporter_id VARCHAR(50),
    fraud_type VARCHAR(50),
    estimated_loss DECIMAL(15,2),
    evidence JSONB,
    tags TEXT[],
    sla_deadline TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_cdr_caller_id ON cdr_records(caller_id);
CREATE INDEX IF NOT EXISTS idx_cdr_timestamp ON cdr_records(start_time);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_user_id ON fraud_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_timestamp ON fraud_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_severity ON fraud_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_status ON fraud_cases(status);
CREATE INDEX IF NOT EXISTS idx_fraud_cases_priority ON fraud_cases(priority);

-- Insert sample data
INSERT INTO user_profiles (user_id, risk_level) VALUES 
('user001', 'LOW'),
('user002', 'MEDIUM'),
('user003', 'HIGH')
ON CONFLICT (user_id) DO NOTHING;
"

print_success "Database schema initialized"

# Submit Flink job
print_status "Submitting Flink fraud detection job..."
kubectl exec -n $NAMESPACE deployment/flink-jobmanager -- /opt/flink/bin/flink run \
    --class jobs.EnhancedFraudDetectionJob \
    --parallelism 2 \
    /opt/flink/examples/fraudguard-flink-jobs.jar || print_warning "Flink job submission may have failed"

# Deploy sample data generator
print_status "Deploying sample data generator..."
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: data-generator
  namespace: $NAMESPACE
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: data-generator
            image: python:3.9-slim
            command:
            - /bin/bash
            - -c
            - |
              pip install requests faker
              python -c "
              import requests
              import json
              import random
              from faker import Faker
              from datetime import datetime, timedelta
              
              fake = Faker()
              
              for i in range(10):
                  cdr = {
                      'id': f'cdr_{i:06d}',
                      'caller_id': f'user{random.randint(1,100):03d}',
                      'callee_id': fake.phone_number(),
                      'call_type': random.choice(['LOCAL', 'NATIONAL', 'INTERNATIONAL', 'ROAMING']),
                      'start_time': (datetime.now() - timedelta(minutes=random.randint(1,60))).isoformat(),
                      'duration': random.randint(30, 1800),
                      'cost': round(random.uniform(0.1, 100.0), 2),
                      'location_caller': fake.city(),
                      'device_imei': fake.numerify('###############'),
                      'bytes_transmitted': random.randint(1000, 10000000)
                  }
                  
                  try:
                      response = requests.post('http://api-gateway.${NAMESPACE}.svc.cluster.local:8000/api/ingest/cdr', 
                                             json=cdr, timeout=10)
                      print(f'Sent CDR {i}: {response.status_code}')
                  except Exception as e:
                      print(f'Error sending CDR {i}: {e}')
              "
          restartPolicy: OnFailure
EOF

print_success "Sample data generator deployed"

# Get service information
print_status "Gathering service information..."

# Get external IPs
FRONTEND_IP=$(kubectl get service frontend -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
GRAFANA_IP=$(kubectl get service prometheus-grafana -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Use port-forward")

# Display deployment summary
print_success "FraudGuard 360 deployment completed successfully!"
echo ""
echo "=== DEPLOYMENT SUMMARY ==="
echo ""
echo "Namespace: $NAMESPACE"
echo ""
echo "Application URLs:"
echo "  Frontend: http://$FRONTEND_IP (or use port-forward: kubectl port-forward service/frontend 3000:80 -n $NAMESPACE)"
echo "  API Gateway: kubectl port-forward service/api-gateway 8000:8000 -n $NAMESPACE"
echo "  ML Service: kubectl port-forward service/ml-service 8005:8005 -n $NAMESPACE"
echo "  Flink UI: kubectl port-forward service/flink-jobmanager 8081:8081 -n $NAMESPACE"
echo ""
echo "Monitoring:"
echo "  Grafana: kubectl port-forward service/prometheus-grafana 3001:80 -n $NAMESPACE"
echo "  Grafana Login: admin / fraudguard123"
echo "  Prometheus: kubectl port-forward service/prometheus-kube-prometheus-prometheus 9090:9090 -n $NAMESPACE"
echo ""
echo "Databases:"
echo "  PostgreSQL: kubectl port-forward service/postgres 5432:5432 -n $NAMESPACE"
echo "  PostgreSQL Login: postgres / fraudguard123"
echo "  Neo4j: kubectl port-forward service/neo4j 7474:7474 -n $NAMESPACE"
echo "  Neo4j Login: neo4j / fraudguard123"
echo "  Redis: kubectl port-forward service/redis-master 6379:6379 -n $NAMESPACE"
echo ""
echo "Message Queue:"
echo "  Kafka: kubectl port-forward service/kafka 9092:9092 -n $NAMESPACE"
echo ""
echo "To access the application:"
echo "1. Add '127.0.0.1 fraudguard.local' to your /etc/hosts file"
echo "2. Visit http://fraudguard.local"
echo ""
echo "To check deployment status:"
echo "kubectl get pods -n $NAMESPACE"
echo ""
echo "To view logs:"
echo "kubectl logs -f deployment/api-gateway -n $NAMESPACE"
echo ""
echo "To clean up deployment:"
echo "kubectl delete namespace $NAMESPACE"
echo ""

# Save deployment info to file
cat <<EOF > deployment-info.txt
FraudGuard 360 Deployment Information
Generated: $(date)

Namespace: $NAMESPACE

Port Forward Commands:
kubectl port-forward service/frontend 3000:80 -n $NAMESPACE
kubectl port-forward service/api-gateway 8000:8000 -n $NAMESPACE
kubectl port-forward service/ml-service 8005:8005 -n $NAMESPACE
kubectl port-forward service/flink-jobmanager 8081:8081 -n $NAMESPACE
kubectl port-forward service/prometheus-grafana 3001:80 -n $NAMESPACE
kubectl port-forward service/postgres 5432:5432 -n $NAMESPACE
kubectl port-forward service/neo4j 7474:7474 -n $NAMESPACE
kubectl port-forward service/redis-master 6379:6379 -n $NAMESPACE
kubectl port-forward service/kafka 9092:9092 -n $NAMESPACE

Default Credentials:
- Grafana: admin / fraudguard123
- PostgreSQL: postgres / fraudguard123
- Neo4j: neo4j / fraudguard123
- Redis: fraudguard123

Cleanup Command:
kubectl delete namespace $NAMESPACE
EOF

print_success "Deployment information saved to deployment-info.txt"
print_success "FraudGuard 360 is now ready for use!"