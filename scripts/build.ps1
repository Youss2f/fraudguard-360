# FraudGuard 360 - Enterprise Build Script
# Builds all microservices for the fraud detection platform

Write-Host "Building FraudGuard 360 Enterprise Services..." -ForegroundColor Green

# Build Core ML Service
Write-Host "Building Core ML Service..." -ForegroundColor Cyan
docker build -t fraudguard-360/core-ml-service:latest ./core-ml-service/

# Build Risk Scoring Service  
Write-Host "Building Risk Scoring Service..." -ForegroundColor Cyan
docker build -t fraudguard-360/risk-scoring-service:latest ./risk-scoring-service/

# Build Graph Analytics Service
Write-Host "Building Graph Analytics Service..." -ForegroundColor Cyan
docker build -t fraudguard-360/graph-analytics-service:latest ./graph-analytics-service/

# Build API Gateway
Write-Host "Building API Gateway..." -ForegroundColor Cyan
docker build -t fraudguard-360/api-gateway:latest ./api-gateway/

# Build Frontend
Write-Host "Building Frontend Dashboard..." -ForegroundColor Cyan
docker build -t fraudguard-360/frontend:latest ./frontend/

# Build Flink Job
Write-Host "Building Flink Stream Processor..." -ForegroundColor Cyan
Set-Location stream-processor-flink
mvn clean package -DskipTests
Set-Location ..

Write-Host "All services built successfully!" -ForegroundColor Green
Write-Host "Run 'docker-compose up -d' to start the platform." -ForegroundColor Yellow