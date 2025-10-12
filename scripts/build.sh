#!/bin/bash

# FraudGuard 360 - Enterprise Build Script
# Builds all microservices for the fraud detection platform

set -e

echo "Building FraudGuard 360 Enterprise Services..."

# Build Core ML Service
echo "Building Core ML Service..."
docker build -t fraudguard-360/core-ml-service:latest ./core-ml-service/

# Build Risk Scoring Service  
echo "Building Risk Scoring Service..."
docker build -t fraudguard-360/risk-scoring-service:latest ./risk-scoring-service/

# Build Graph Analytics Service
echo "Building Graph Analytics Service..."
docker build -t fraudguard-360/graph-analytics-service:latest ./graph-analytics-service/

# Build API Gateway
echo "Building API Gateway..."
docker build -t fraudguard-360/api-gateway:latest ./api-gateway/

# Build Frontend
echo "Building Frontend Dashboard..."
docker build -t fraudguard-360/frontend:latest ./frontend/

# Build Flink Job
echo "Building Flink Stream Processor..."
cd stream-processor-flink
mvn clean package -DskipTests
cd ..

echo "All services built successfully!"
echo "Run 'docker-compose up -d' to start the platform."