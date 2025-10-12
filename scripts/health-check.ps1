# FraudGuard 360° - Deployment Health Check
# Verifies all enterprise services are running properly

Write-Host "FraudGuard 360° Enterprise Health Check" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

$services = @(
    @{Name="API Gateway"; Port=8000; Path="/health"}
    @{Name="Core ML Service"; Port=8001; Path="/health"}
    @{Name="Risk Scoring Service"; Port=8002; Path="/health"}
    @{Name="Graph Analytics Service"; Port=8003; Path="/health"}
    @{Name="Frontend Dashboard"; Port=3000; Path="/"}
)

$allHealthy = $true

foreach ($service in $services) {
    Write-Host "Checking $($service.Name)..." -ForegroundColor Cyan
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$($service.Port)$($service.Path)" -Method GET -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ $($service.Name) - Healthy" -ForegroundColor Green
        } else {
            Write-Host "⚠️ $($service.Name) - Status: $($response.StatusCode)" -ForegroundColor Yellow
            $allHealthy = $false
        }
    }
    catch {
        Write-Host "❌ $($service.Name) - Not responding" -ForegroundColor Red
        $allHealthy = $false
    }
}

Write-Host ""
if ($allHealthy) {
    Write-Host "🎉 All services are healthy! FraudGuard 360° is ready for enterprise fraud detection." -ForegroundColor Green
} else {
    Write-Host "⚠️ Some services need attention. Check Docker containers: 'docker-compose ps'" -ForegroundColor Yellow
}

# Check Docker services
Write-Host ""
Write-Host "Docker Services Status:" -ForegroundColor Cyan
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"