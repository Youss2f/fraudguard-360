# Script PowerShell pour créer un déploiement Kubernetes temporaire pour screenshots
# Usage: .\create_k8s_demo.ps1

Write-Host "🚀 Création du namespace fraudguard-360..." -ForegroundColor Green

# Créer le namespace
kubectl create namespace fraudguard-360 --dry-run=client -o yaml | kubectl apply -f -

Write-Host "📦 Déploiement des composants FraudGuard 360°..." -ForegroundColor Yellow

# API Gateway
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraudguard-api-gateway
  namespace: fraudguard-360
  labels:
    app: api-gateway
    component: fraudguard-360
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      labels:
        app: api-gateway
        component: fraudguard-360
    spec:
      containers:
      - name: api-gateway
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "250m"
        env:
        - name: SERVICE_NAME
          value: "fraudguard-api-gateway"
        - name: VERSION
          value: "v1.3.0"
---
apiVersion: v1
kind: Service
metadata:
  name: fraudguard-api-gateway
  namespace: fraudguard-360
spec:
  selector:
    app: api-gateway
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
"@ | kubectl apply -f -

# ML Service
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraudguard-ml-service
  namespace: fraudguard-360
  labels:
    app: ml-service
    component: fraudguard-360
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
        component: fraudguard-360
    spec:
      containers:
      - name: ml-service
        image: python:3.9-slim
        command: ["python", "-c", "import time; time.sleep(3600)"]
        ports:
        - containerPort: 8001
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        env:
        - name: SERVICE_NAME
          value: "fraudguard-ml-service"
---
apiVersion: v1
kind: Service
metadata:
  name: fraudguard-ml-service
  namespace: fraudguard-360
spec:
  selector:
    app: ml-service
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
"@ | kubectl apply -f -

# Frontend
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraudguard-frontend
  namespace: fraudguard-360
  labels:
    app: frontend
    component: fraudguard-360
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
        component: fraudguard-360
    spec:
      containers:
      - name: frontend
        image: nginx:1.21-alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "50m"
          limits:
            memory: "256Mi"
            cpu: "100m"
        env:
        - name: SERVICE_NAME
          value: "fraudguard-frontend"
---
apiVersion: v1
kind: Service
metadata:
  name: fraudguard-frontend
  namespace: fraudguard-360
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
"@ | kubectl apply -f -

# Flink JobManager
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-jobmanager
  namespace: fraudguard-360
  labels:
    app: flink-jobmanager
    component: fraudguard-360
spec:
  replicas: 1
  selector:
    matchLabels:
      app: flink-jobmanager
  template:
    metadata:
      labels:
        app: flink-jobmanager
        component: fraudguard-360
    spec:
      containers:
      - name: jobmanager
        image: flink:1.17-scala_2.12
        args: ["jobmanager"]
        ports:
        - containerPort: 8081
        - containerPort: 6123
        resources:
          requests:
            memory: "1Gi"
            cpu: "200m"
          limits:
            memory: "2Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: flink-jobmanager
  namespace: fraudguard-360
spec:
  selector:
    app: flink-jobmanager
  ports:
  - name: rpc
    port: 6123
    targetPort: 6123
  - name: web
    port: 8081
    targetPort: 8081
  type: ClusterIP
"@ | kubectl apply -f -

# Flink TaskManager
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flink-taskmanager
  namespace: fraudguard-360
  labels:
    app: flink-taskmanager
    component: fraudguard-360
spec:
  replicas: 2
  selector:
    matchLabels:
      app: flink-taskmanager
  template:
    metadata:
      labels:
        app: flink-taskmanager
        component: fraudguard-360
    spec:
      containers:
      - name: taskmanager
        image: flink:1.17-scala_2.12
        args: ["taskmanager"]
        ports:
        - containerPort: 6122
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1"
"@ | kubectl apply -f -

# Kafka
@"
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: kafka
  namespace: fraudguard-360
  labels:
    app: kafka
    component: fraudguard-360
spec:
  serviceName: kafka-headless
  replicas: 3
  selector:
    matchLabels:
      app: kafka
  template:
    metadata:
      labels:
        app: kafka
        component: fraudguard-360
    spec:
      containers:
      - name: kafka
        image: confluentinc/cp-kafka:7.4.0
        ports:
        - containerPort: 9092
        resources:
          requests:
            memory: "1Gi"
            cpu: "300m"
          limits:
            memory: "2Gi"
            cpu: "500m"
        env:
        - name: KAFKA_ZOOKEEPER_CONNECT
          value: "zookeeper:2181"
        - name: KAFKA_ADVERTISED_LISTENERS
          value: "PLAINTEXT://localhost:9092"
        - name: KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR
          value: "3"
---
apiVersion: v1
kind: Service
metadata:
  name: kafka-headless
  namespace: fraudguard-360
spec:
  clusterIP: None
  selector:
    app: kafka
  ports:
  - port: 9092
    targetPort: 9092
"@ | kubectl apply -f -

# Neo4j
@"
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neo4j-core
  namespace: fraudguard-360
  labels:
    app: neo4j
    component: fraudguard-360
spec:
  serviceName: neo4j-admin
  replicas: 3
  selector:
    matchLabels:
      app: neo4j
  template:
    metadata:
      labels:
        app: neo4j
        component: fraudguard-360
    spec:
      containers:
      - name: neo4j
        image: neo4j:5.11-community
        ports:
        - containerPort: 7474
        - containerPort: 7687
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1"
        env:
        - name: NEO4J_AUTH
          value: "neo4j/fraudguard360"
---
apiVersion: v1
kind: Service
metadata:
  name: neo4j-admin
  namespace: fraudguard-360
spec:
  selector:
    app: neo4j
  ports:
  - name: http
    port: 7474
    targetPort: 7474
  - name: bolt
    port: 7687
    targetPort: 7687
  type: ClusterIP
"@ | kubectl apply -f -

# Prometheus
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus-server
  namespace: fraudguard-360
  labels:
    app: prometheus
    component: fraudguard-360
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
        component: fraudguard-360
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
        resources:
          requests:
            memory: "512Mi"
            cpu: "100m"
          limits:
            memory: "1Gi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus-server
  namespace: fraudguard-360
spec:
  selector:
    app: prometheus
  ports:
  - port: 9090
    targetPort: 9090
  type: ClusterIP
"@ | kubectl apply -f -

# Grafana
@"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana-monitoring
  namespace: fraudguard-360
  labels:
    app: grafana
    component: fraudguard-360
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
        component: fraudguard-360
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.0.3
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "fraudguard360"
---
apiVersion: v1
kind: Service
metadata:
  name: grafana-monitoring
  namespace: fraudguard-360
spec:
  selector:
    app: grafana
  ports:
  - port: 3000
    targetPort: 3000
  type: LoadBalancer
"@ | kubectl apply -f -

Write-Host "⏳ Attente du démarrage des pods..." -ForegroundColor Yellow
kubectl wait --for=condition=Ready pod -l component=fraudguard-360 -n fraudguard-360 --timeout=300s

Write-Host ""
Write-Host "🎯 Déploiement terminé ! Exécutez maintenant:" -ForegroundColor Green
Write-Host ""
Write-Host "# Pour la Figure 4.1 (screenshot des pods):" -ForegroundColor Cyan
Write-Host "kubectl get pods -n fraudguard-360 -o wide" -ForegroundColor White
Write-Host ""
Write-Host "# Pour vérifier les services:" -ForegroundColor Cyan
Write-Host "kubectl get services -n fraudguard-360" -ForegroundColor White
Write-Host ""
Write-Host "# Pour les métriques de performance:" -ForegroundColor Cyan
Write-Host "kubectl top pods -n fraudguard-360" -ForegroundColor White
Write-Host ""
Write-Host "# Pour nettoyer après les screenshots:" -ForegroundColor Cyan
Write-Host "kubectl delete namespace fraudguard-360" -ForegroundColor White
Write-Host ""