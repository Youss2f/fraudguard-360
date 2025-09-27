#!/bin/bash

# FraudGuard 360 - Complete Build and Deployment Script
# This script builds all components, runs tests, and deploys the system
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="fraudguard-360"
VERSION="${VERSION:-1.0.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Default deployment mode
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-development}"
SKIP_TESTS="${SKIP_TESTS:-false}"
PUSH_IMAGES="${PUSH_IMAGES:-false}"
REGISTRY="${REGISTRY:-localhost:5000}"

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

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check Java (for Flink jobs)
    if ! command -v java &> /dev/null; then
        print_error "Java is not installed or not in PATH"
        exit 1
    fi
    
    # Check Maven (for Flink jobs)
    if ! command -v mvn &> /dev/null; then
        print_error "Maven is not installed or not in PATH"
        exit 1
    fi
    
    # Check Node.js (for frontend)
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed or not in PATH"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed or not in PATH"
        exit 1
    fi
    
    # Check kubectl (optional, for Kubernetes deployment)
    if command -v kubectl &> /dev/null; then
        print_status "kubectl found - Kubernetes deployment available"
    else
        print_warning "kubectl not found - Kubernetes deployment will be skipped"
    fi
    
    print_success "Prerequisites check completed"
}

# Function to build Java components
build_java_components() {
    print_status "Building Java components (Flink jobs)..."
    
    cd flink-jobs
    
    # Clean and build
    mvn clean compile
    mvn test -Dtest=* || {
        if [[ "${SKIP_TESTS}" != "true" ]]; then
            print_error "Java tests failed"
            exit 1
        else
            print_warning "Java tests failed but continuing due to SKIP_TESTS=true"
        fi
    }
    
    # Package
    mvn package -DskipTests=true
    
    cd ..
    print_success "Java components built successfully"
}

# Function to build Docker images
build_docker_images() {
    print_status "Building Docker images..."
    
    # Build API Gateway
    print_status "Building API Gateway image..."
    docker build \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${GIT_COMMIT}" \
        --build-arg VERSION="${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}-api-gateway:${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}-api-gateway:latest" \
        ./api-gateway
    
    # Build ML Service
    print_status "Building ML Service image..."
    docker build \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${GIT_COMMIT}" \
        --build-arg VERSION="${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}-ml-service:${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}-ml-service:latest" \
        ./ml-service
    
    # Build Frontend
    print_status "Building Frontend image..."
    docker build \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VCS_REF="${GIT_COMMIT}" \
        --build-arg VERSION="${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}-frontend:${VERSION}" \
        -t "${REGISTRY}/${PROJECT_NAME}-frontend:latest" \
        ./frontend
    
    print_success "Docker images built successfully"
}

# Function to run component tests
run_tests() {
    if [[ "${SKIP_TESTS}" == "true" ]]; then
        print_warning "Skipping tests due to SKIP_TESTS=true"
        return
    fi
    
    print_status "Running component tests..."
    
    # Test API Gateway
    print_status "Testing API Gateway..."
    cd api-gateway
    python -m pytest tests/ -v || {
        print_error "API Gateway tests failed"
        cd ..
        exit 1
    }
    cd ..
    
    # Test ML Service
    print_status "Testing ML Service..."
    cd ml-service
    python -m pytest tests/ -v || {
        print_error "ML Service tests failed"
        cd ..
        exit 1
    }
    cd ..
    
    # Test Frontend
    print_status "Testing Frontend..."
    cd frontend
    npm test -- --coverage --watchAll=false || {
        print_error "Frontend tests failed"
        cd ..
        exit 1
    }
    cd ..
    
    print_success "All component tests passed"
}

# Function to run integration tests
run_integration_tests() {
    if [[ "${SKIP_TESTS}" == "true" ]]; then
        print_warning "Skipping integration tests due to SKIP_TESTS=true"
        return
    fi
    
    print_status "Running integration tests..."
    
    # Start services for integration testing
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Run integration tests
    python tests/integration/test_fraud_detection_pipeline.py || {
        print_error "Integration tests failed"
        docker-compose down
        exit 1
    }
    
    # Cleanup
    docker-compose down
    
    print_success "Integration tests passed"
}

# Function to push Docker images
push_images() {
    if [[ "${PUSH_IMAGES}" != "true" ]]; then
        print_status "Skipping image push (PUSH_IMAGES=false)"
        return
    fi
    
    print_status "Pushing Docker images to registry..."
    
    docker push "${REGISTRY}/${PROJECT_NAME}-api-gateway:${VERSION}"
    docker push "${REGISTRY}/${PROJECT_NAME}-api-gateway:latest"
    docker push "${REGISTRY}/${PROJECT_NAME}-ml-service:${VERSION}"
    docker push "${REGISTRY}/${PROJECT_NAME}-ml-service:latest"
    docker push "${REGISTRY}/${PROJECT_NAME}-frontend:${VERSION}"
    docker push "${REGISTRY}/${PROJECT_NAME}-frontend:latest"
    
    print_success "Images pushed successfully"
}

# Function to deploy with Docker Compose
deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."
    
    case "${DEPLOYMENT_MODE}" in
        "development")
            docker-compose -f docker-compose.yml up -d
            ;;
        "production")
            docker-compose -f docker-compose.production.yml up -d
            ;;
        *)
            print_error "Unknown deployment mode: ${DEPLOYMENT_MODE}"
            exit 1
            ;;
    esac
    
    print_success "Docker Compose deployment completed"
}

# Function to deploy to Kubernetes
deploy_kubernetes() {
    if ! command -v kubectl &> /dev/null; then
        print_warning "kubectl not found - skipping Kubernetes deployment"
        return
    fi
    
    print_status "Deploying to Kubernetes..."
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmaps/
    kubectl apply -f k8s/secrets/
    kubectl apply -f k8s/services/
    kubectl apply -f k8s/deployments/
    kubectl apply -f k8s/ingress/
    
    # Wait for rollout
    kubectl rollout status deployment/api-gateway -n fraudguard-360
    kubectl rollout status deployment/ml-service -n fraudguard-360
    kubectl rollout status deployment/frontend -n fraudguard-360
    
    print_success "Kubernetes deployment completed"
}

# Function to run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Wait a bit for services to start
    sleep 10
    
    # Check API Gateway
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "API Gateway health check passed"
    else
        print_error "API Gateway health check failed"
        return 1
    fi
    
    # Check ML Service
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        print_success "ML Service health check passed"
    else
        print_error "ML Service health check failed"
        return 1
    fi
    
    # Check Frontend
    if curl -f http://localhost:3000 >/dev/null 2>&1; then
        print_success "Frontend health check passed"
    else
        print_error "Frontend health check failed"
        return 1
    fi
    
    print_success "All health checks passed"
}

# Function to generate deployment report
generate_report() {
    print_status "Generating deployment report..."
    
    cat > deployment-report.md << EOF
# FraudGuard 360 Deployment Report

**Deployment Date:** $(date)
**Version:** ${VERSION}
**Git Commit:** ${GIT_COMMIT}
**Deployment Mode:** ${DEPLOYMENT_MODE}

## Components Built

- ✅ API Gateway (${REGISTRY}/${PROJECT_NAME}-api-gateway:${VERSION})
- ✅ ML Service (${REGISTRY}/${PROJECT_NAME}-ml-service:${VERSION})
- ✅ Frontend (${REGISTRY}/${PROJECT_NAME}-frontend:${VERSION})
- ✅ Flink Jobs (JAR: flink-jobs-${VERSION}.jar)

## Service URLs

- **Frontend Dashboard:** http://localhost:3000
- **API Gateway:** http://localhost:8000
- **ML Service:** http://localhost:8001
- **Neo4j Browser:** http://localhost:7474
- **Flink Web UI:** http://localhost:8081

## Quick Start Commands

\`\`\`bash
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down

# View metrics
curl http://localhost:8000/metrics
\`\`\`

## Next Steps

1. Access the FraudGuard dashboard at http://localhost:3000
2. Configure data sources and alert thresholds
3. Monitor system health and performance metrics
4. Review fraud detection alerts and patterns

EOF

    print_success "Deployment report generated: deployment-report.md"
}

# Main deployment function
main() {
    print_status "Starting FraudGuard 360 build and deployment..."
    print_status "Version: ${VERSION}"
    print_status "Deployment Mode: ${DEPLOYMENT_MODE}"
    print_status "Build Date: ${BUILD_DATE}"
    print_status "Git Commit: ${GIT_COMMIT}"
    echo
    
    # Run all steps
    check_prerequisites
    build_java_components
    build_docker_images
    run_tests
    push_images
    
    # Deploy based on mode
    case "${DEPLOYMENT_MODE}" in
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
        *)
            deploy_docker_compose
            ;;
    esac
    
    run_integration_tests
    run_health_checks
    generate_report
    
    print_success "🎉 FraudGuard 360 deployment completed successfully!"
    print_status "Access the dashboard at: http://localhost:3000"
    echo
}

# Handle script arguments
case "${1:-}" in
    "build")
        check_prerequisites
        build_java_components
        build_docker_images
        ;;
    "test")
        run_tests
        run_integration_tests
        ;;
    "deploy")
        deploy_docker_compose
        ;;
    "k8s")
        deploy_kubernetes
        ;;
    "health")
        run_health_checks
        ;;
    "report")
        generate_report
        ;;
    "help"|"-h"|"--help")
        echo "FraudGuard 360 Build and Deployment Script"
        echo
        echo "Usage: $0 [command]"
        echo
        echo "Commands:"
        echo "  build     Build all components"
        echo "  test      Run tests"
        echo "  deploy    Deploy with Docker Compose"
        echo "  k8s       Deploy to Kubernetes"
        echo "  health    Run health checks"
        echo "  report    Generate deployment report"
        echo "  help      Show this help message"
        echo
        echo "Environment Variables:"
        echo "  VERSION              Version tag (default: 1.0.0)"
        echo "  DEPLOYMENT_MODE      deployment|production|kubernetes (default: development)"
        echo "  SKIP_TESTS          Skip tests (default: false)"
        echo "  PUSH_IMAGES         Push images to registry (default: false)"
        echo "  REGISTRY            Docker registry (default: localhost:5000)"
        echo
        echo "Examples:"
        echo "  $0                                    # Full build and deploy"
        echo "  VERSION=2.0.0 $0 build              # Build specific version"
        echo "  DEPLOYMENT_MODE=production $0        # Production deployment"
        echo "  SKIP_TESTS=true $0 deploy           # Deploy without tests"
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac