# FraudGuard 360 - PowerShell Build and Deployment Script
# Complete build, test, and deployment automation for Windows

param(
    [string]$Command = "",
    [string]$Version = "1.0.0",
    [string]$DeploymentMode = "development",
    [string]$Registry = "localhost:5000",
    [switch]$SkipTests = $false,
    [switch]$PushImages = $false
)

# Color functions for output
function Write-Status { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Blue }
function Write-Success { param($Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

# Configuration
$ProjectName = "fraudguard-360"
$BuildDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$GitCommit = try { (git rev-parse --short HEAD 2>$null) } catch { "unknown" }

function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    $Prerequisites = @(
        @{Name = "Docker"; Command = "docker --version"},
        @{Name = "Docker Compose"; Command = "docker-compose --version"},
        @{Name = "Java"; Command = "java -version"},
        @{Name = "Maven"; Command = "mvn --version"},
        @{Name = "Node.js"; Command = "node --version"},
        @{Name = "npm"; Command = "npm --version"}
    )
    
    foreach ($prereq in $Prerequisites) {
        try {
            Invoke-Expression $prereq.Command | Out-Null
            Write-Status "$($prereq.Name) is available"
        }
        catch {
            Write-Error "$($prereq.Name) is not installed or not in PATH"
            return $false
        }
    }
    
    # Check optional kubectl
    try {
        kubectl version --client | Out-Null
        Write-Status "kubectl found - Kubernetes deployment available"
    }
    catch {
        Write-Warning "kubectl not found - Kubernetes deployment will be skipped"
    }
    
    Write-Success "Prerequisites check completed"
    return $true
}

function Build-JavaComponents {
    Write-Status "Building Java components (Flink jobs)..."
    
    Push-Location "flink-jobs"
    
    try {
        # Clean and build
        mvn clean compile
        if ($LASTEXITCODE -ne 0) { throw "Maven compile failed" }
        
        if (-not $SkipTests) {
            mvn test
            if ($LASTEXITCODE -ne 0) {
                if ($SkipTests) {
                    Write-Warning "Java tests failed but continuing due to SkipTests flag"
                } else {
                    throw "Java tests failed"
                }
            }
        }
        
        # Package
        mvn package -DskipTests=true
        if ($LASTEXITCODE -ne 0) { throw "Maven package failed" }
        
        Write-Success "Java components built successfully"
    }
    catch {
        Write-Error "Java build failed: $_"
        Pop-Location
        return $false
    }
    finally {
        Pop-Location
    }
    
    return $true
}

function Build-DockerImages {
    Write-Status "Building Docker images..."
    
    $Images = @(
        @{Name = "API Gateway"; Path = "./api-gateway"; Tag = "api-gateway"},
        @{Name = "ML Service"; Path = "./ml-service"; Tag = "ml-service"},
        @{Name = "Frontend"; Path = "./frontend"; Tag = "frontend"}
    )
    
    foreach ($image in $Images) {
        Write-Status "Building $($image.Name) image..."
        
        $buildArgs = @(
            "--build-arg BUILD_DATE=$BuildDate",
            "--build-arg VCS_REF=$GitCommit",
            "--build-arg VERSION=$Version",
            "-t $Registry/$ProjectName-$($image.Tag):$Version",
            "-t $Registry/$ProjectName-$($image.Tag):latest",
            $image.Path
        )
        
        docker build @buildArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build $($image.Name) image"
            return $false
        }
    }
    
    Write-Success "Docker images built successfully"
    return $true
}

function Test-Components {
    if ($SkipTests) {
        Write-Warning "Skipping tests due to SkipTests flag"
        return $true
    }
    
    Write-Status "Running component tests..."
    
    # Test API Gateway
    Write-Status "Testing API Gateway..."
    Push-Location "api-gateway"
    try {
        python -m pytest tests/ -v
        if ($LASTEXITCODE -ne 0) { throw "API Gateway tests failed" }
    }
    catch {
        Write-Error "API Gateway tests failed: $_"
        Pop-Location
        return $false
    }
    finally {
        Pop-Location
    }
    
    # Test ML Service
    Write-Status "Testing ML Service..."
    Push-Location "ml-service"
    try {
        python -m pytest tests/ -v
        if ($LASTEXITCODE -ne 0) { throw "ML Service tests failed" }
    }
    catch {
        Write-Error "ML Service tests failed: $_"
        Pop-Location
        return $false
    }
    finally {
        Pop-Location
    }
    
    # Test Frontend
    Write-Status "Testing Frontend..."
    Push-Location "frontend"
    try {
        npm test -- --coverage --watchAll=false
        if ($LASTEXITCODE -ne 0) { throw "Frontend tests failed" }
    }
    catch {
        Write-Error "Frontend tests failed: $_"
        Pop-Location
        return $false
    }
    finally {
        Pop-Location
    }
    
    Write-Success "All component tests passed"
    return $true
}

function Test-Integration {
    if ($SkipTests) {
        Write-Warning "Skipping integration tests due to SkipTests flag"
        return $true
    }
    
    Write-Status "Running integration tests..."
    
    # Start services for integration testing
    docker-compose -f docker-compose.yml up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start services for integration testing"
        return $false
    }
    
    # Wait for services to be ready
    Write-Status "Waiting for services to be ready..."
    Start-Sleep -Seconds 30
    
    try {
        # Run integration tests
        python tests/comprehensive_test_suite.py
        $testResult = $LASTEXITCODE -eq 0
        
        if ($testResult) {
            Write-Success "Integration tests passed"
        } else {
            Write-Error "Integration tests failed"
        }
    }
    catch {
        Write-Error "Integration tests failed: $_"
        $testResult = $false
    }
    finally {
        # Cleanup
        docker-compose down
    }
    
    return $testResult
}

function Push-Images {
    if (-not $PushImages) {
        Write-Status "Skipping image push (PushImages = false)"
        return $true
    }
    
    Write-Status "Pushing Docker images to registry..."
    
    $Tags = @(
        "$Registry/$ProjectName-api-gateway:$Version",
        "$Registry/$ProjectName-api-gateway:latest",
        "$Registry/$ProjectName-ml-service:$Version",
        "$Registry/$ProjectName-ml-service:latest",
        "$Registry/$ProjectName-frontend:$Version",
        "$Registry/$ProjectName-frontend:latest"
    )
    
    foreach ($tag in $Tags) {
        docker push $tag
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to push image: $tag"
            return $false
        }
    }
    
    Write-Success "Images pushed successfully"
    return $true
}

function Deploy-DockerCompose {
    Write-Status "Deploying with Docker Compose..."
    
    $composeFile = switch ($DeploymentMode) {
        "development" { "docker-compose.yml" }
        "production" { "docker-compose.production.yml" }
        default {
            Write-Error "Unknown deployment mode: $DeploymentMode"
            return $false
        }
    }
    
    docker-compose -f $composeFile up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker Compose deployment failed"
        return $false
    }
    
    Write-Success "Docker Compose deployment completed"
    return $true
}

function Deploy-Kubernetes {
    $kubectlAvailable = try { kubectl version --client | Out-Null; $true } catch { $false }
    
    if (-not $kubectlAvailable) {
        Write-Warning "kubectl not found - skipping Kubernetes deployment"
        return $true
    }
    
    Write-Status "Deploying to Kubernetes..."
    
    $manifests = @(
        "k8s/namespace.yaml",
        "k8s/configmaps/",
        "k8s/secrets/",
        "k8s/services/",
        "k8s/deployments/",
        "k8s/ingress/"
    )
    
    foreach ($manifest in $manifests) {
        kubectl apply -f $manifest
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to apply manifest: $manifest"
            return $false
        }
    }
    
    # Wait for rollout
    $deployments = @("api-gateway", "ml-service", "frontend")
    foreach ($deployment in $deployments) {
        kubectl rollout status deployment/$deployment -n fraudguard-360
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Rollout failed for deployment: $deployment"
            return $false
        }
    }
    
    Write-Success "Kubernetes deployment completed"
    return $true
}

function Test-HealthChecks {
    Write-Status "Running health checks..."
    
    # Wait a bit for services to start
    Start-Sleep -Seconds 10
    
    $Services = @(
        @{Name = "API Gateway"; Url = "http://localhost:8000/health"},
        @{Name = "ML Service"; Url = "http://localhost:8001/health"},
        @{Name = "Frontend"; Url = "http://localhost:3000"}
    )
    
    $allHealthy = $true
    
    foreach ($service in $Services) {
        try {
            $response = Invoke-WebRequest -Uri $service.Url -TimeoutSec 10 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Success "$($service.Name) health check passed"
            } else {
                Write-Error "$($service.Name) health check failed: Status $($response.StatusCode)"
                $allHealthy = $false
            }
        }
        catch {
            Write-Error "$($service.Name) health check failed: $_"
            $allHealthy = $false
        }
    }
    
    if ($allHealthy) {
        Write-Success "All health checks passed"
    } else {
        Write-Error "Some health checks failed"
    }
    
    return $allHealthy
}

function New-DeploymentReport {
    Write-Status "Generating deployment report..."
    
    $reportContent = @"
# FraudGuard 360 Deployment Report

**Deployment Date:** $(Get-Date)
**Version:** $Version
**Git Commit:** $GitCommit
**Deployment Mode:** $DeploymentMode

## Components Built

- ✅ API Gateway ($Registry/$ProjectName-api-gateway:$Version)
- ✅ ML Service ($Registry/$ProjectName-ml-service:$Version)
- ✅ Frontend ($Registry/$ProjectName-frontend:$Version)
- ✅ Flink Jobs (JAR: flink-jobs-$Version.jar)

## Service URLs

- **Frontend Dashboard:** http://localhost:3000
- **API Gateway:** http://localhost:8000
- **ML Service:** http://localhost:8001
- **Neo4j Browser:** http://localhost:7474
- **Flink Web UI:** http://localhost:8081

## Quick Start Commands

``````powershell
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down

# View metrics
Invoke-WebRequest http://localhost:8000/metrics
``````

## Next Steps

1. Access the FraudGuard dashboard at http://localhost:3000
2. Configure data sources and alert thresholds
3. Monitor system health and performance metrics
4. Review fraud detection alerts and patterns

"@

    $reportContent | Out-File -FilePath "deployment-report.md" -Encoding UTF8
    Write-Success "Deployment report generated: deployment-report.md"
}

function Invoke-MainDeployment {
    Write-Status "Starting FraudGuard 360 build and deployment..."
    Write-Status "Version: $Version"
    Write-Status "Deployment Mode: $DeploymentMode"
    Write-Status "Build Date: $BuildDate"
    Write-Status "Git Commit: $GitCommit"
    Write-Host ""
    
    # Run all steps
    if (-not (Test-Prerequisites)) { return $false }
    if (-not (Build-JavaComponents)) { return $false }
    if (-not (Build-DockerImages)) { return $false }
    if (-not (Test-Components)) { return $false }
    if (-not (Push-Images)) { return $false }
    
    # Deploy based on mode
    $deploymentSuccess = switch ($DeploymentMode) {
        "kubernetes" { Deploy-Kubernetes }
        "k8s" { Deploy-Kubernetes }
        default { Deploy-DockerCompose }
    }
    
    if (-not $deploymentSuccess) { return $false }
    
    if (-not (Test-Integration)) { return $false }
    if (-not (Test-HealthChecks)) { return $false }
    
    New-DeploymentReport
    
    Write-Success "🎉 FraudGuard 360 deployment completed successfully!"
    Write-Status "Access the dashboard at: http://localhost:3000"
    Write-Host ""
    
    return $true
}

# Main execution based on command
switch ($Command.ToLower()) {
    "build" {
        Test-Prerequisites
        Build-JavaComponents
        Build-DockerImages
    }
    "test" {
        Test-Components
        Test-Integration
    }
    "deploy" {
        Deploy-DockerCompose
    }
    "k8s" {
        Deploy-Kubernetes
    }
    "health" {
        Test-HealthChecks
    }
    "report" {
        New-DeploymentReport
    }
    "help" {
        Write-Host "FraudGuard 360 Build and Deployment Script"
        Write-Host ""
        Write-Host "Usage: .\deploy.ps1 [command] [parameters]"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  build     Build all components"
        Write-Host "  test      Run tests"
        Write-Host "  deploy    Deploy with Docker Compose"
        Write-Host "  k8s       Deploy to Kubernetes"
        Write-Host "  health    Run health checks"
        Write-Host "  report    Generate deployment report"
        Write-Host "  help      Show this help message"
        Write-Host ""
        Write-Host "Parameters:"
        Write-Host "  -Version              Version tag (default: 1.0.0)"
        Write-Host "  -DeploymentMode       development|production|kubernetes (default: development)"
        Write-Host "  -SkipTests           Skip tests"
        Write-Host "  -PushImages          Push images to registry"
        Write-Host "  -Registry            Docker registry (default: localhost:5000)"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\deploy.ps1                                    # Full build and deploy"
        Write-Host "  .\deploy.ps1 build -Version 2.0.0             # Build specific version"
        Write-Host "  .\deploy.ps1 deploy -DeploymentMode production # Production deployment"
        Write-Host "  .\deploy.ps1 test -SkipTests                  # Deploy without tests"
    }
    "" {
        Invoke-MainDeployment
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host "Use '.\deploy.ps1 help' for usage information"
        exit 1
    }
}