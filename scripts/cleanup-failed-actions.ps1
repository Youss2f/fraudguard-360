# GitHub Actions Cleanup Script
# Production-grade automation for workflow run management

[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [string]$GitHubToken = $env:GITHUB_TOKEN,
    
    [Parameter(Mandatory=$false)]
    [string]$GitHubUser = $env:GITHUB_USER,
    
    [Parameter(Mandatory=$false)]
    [string]$GitHubRepository = $env:GITHUB_REPOSITORY,
    
    [Parameter(Mandatory=$false)]
    [switch]$WhatIf,
    
    [Parameter(Mandatory=$false)]
    [ValidateRange(1, 50)]
    [int]$MaxRuns = 30
)

# Input validation and security
if (-not $GitHubToken) {
    Write-Error "GitHub token is required. Set GITHUB_TOKEN environment variable or use -GitHubToken parameter."
    exit 1
}

if (-not $GitHubUser -or -not $GitHubRepository) {
    Write-Error "GitHub user and repository are required. Set environment variables or use parameters."
    exit 1
}

# Sanitize inputs
$GitHubUser = $GitHubUser -replace '[^a-zA-Z0-9\-_]', ''
$GitHubRepository = $GitHubRepository -replace '[^a-zA-Z0-9\-_.]', ''

# Configuration
$uriBase = "https://api.github.com"
$baseHeader = @{
    "Authorization" = "Bearer $GitHubToken"
    "Accept" = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
    "User-Agent" = "GitHub-Actions-Cleanup/1.0"
}

Write-Host "GitHub Actions Cleanup Script" -ForegroundColor Green
Write-Host "Repository: $GitHubUser/$GitHubRepository" -ForegroundColor Yellow
Write-Host "WhatIf Mode: $($WhatIf.IsPresent)" -ForegroundColor Yellow
Write-Host ""

try {
    # Get all workflow runs
    Write-Host "Fetching workflow runs..." -ForegroundColor Cyan
    $runsActiveParams = @{
        Uri     = "$uriBase/repos/$GitHubUser/$GitHubRepository/actions/runs"
        Method  = "Get"
        Headers = $baseHeader
    }
    
    $runsActive = Invoke-RestMethod @runsActiveParams
    Write-Host "Total workflow runs found: $($runsActive.total_count)" -ForegroundColor Yellow
    
    # Filter failed runs
    $actionsFailure = $runsActive.workflow_runs | Where-Object { 
        $_.conclusion -eq "failure" -or $_.conclusion -eq "cancelled" -or $_.conclusion -eq "timed_out"
    }
    
    Write-Host "Failed/Cancelled/Timed-out runs found: $($actionsFailure.Count)" -ForegroundColor Red
    
    if ($actionsFailure.Count -eq 0) {
        Write-Host "No failed workflow runs to clean up!" -ForegroundColor Green
        return
    }
    
    # Display failed runs summary
    Write-Host "`nFailed runs summary:" -ForegroundColor Yellow
    $actionsFailure | ForEach-Object {
        Write-Host "  - ID: $($_.id), Status: $($_.conclusion), Workflow: $($_.name), Branch: $($_.head_branch), Created: $($_.created_at)" -ForegroundColor Gray
    }
    
    if ($WhatIf) {
        Write-Host "`n[WhatIf] Would delete $($actionsFailure.Count) failed workflow runs" -ForegroundColor Magenta
        return
    }
    
    # Confirm deletion
    $confirmation = Read-Host "`nDo you want to delete these $($actionsFailure.Count) failed runs? (y/N)"
    if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
        Write-Host "Operation cancelled." -ForegroundColor Yellow
        return
    }
    
    # Delete failed runs
    Write-Host "`nDeleting failed workflow runs..." -ForegroundColor Red
    $deletedCount = 0
    $errorCount = 0
    
    foreach ($actionFail in $actionsFailure) {
        try {
            $runsDeleteParam = @{
                Uri     = "$uriBase/repos/$GitHubUser/$GitHubRepository/actions/runs/$($actionFail.id)"
                Method  = "Delete"
                Headers = $baseHeader
            }
            
            Write-Host "Deleting run ID: $($actionFail.id) ($($actionFail.name))" -ForegroundColor Yellow
            Invoke-RestMethod @runsDeleteParam
            $deletedCount++
            Start-Sleep -Milliseconds 100  # Rate limiting
        }
        catch {
            Write-Warning "Failed to delete run ID $($actionFail.id): $($_.Exception.Message)"
            $errorCount++
        }
    }
    
    Write-Host "`nCleanup completed!" -ForegroundColor Green
    Write-Host "Successfully deleted: $deletedCount runs" -ForegroundColor Green
    if ($errorCount -gt 0) {
        Write-Host "Failed to delete: $errorCount runs" -ForegroundColor Red
    }
}
catch {
    Write-Error "Error occurred: $($_.Exception.Message)"
    exit 1
}