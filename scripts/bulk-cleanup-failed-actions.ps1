# GitHub Actions Bulk Cleanup Script
# This script handles large-scale cleanup of failed workflow runs with pagination

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubToken,
    
    [Parameter(Mandatory=$false)]
    [string]$GitHubUser = "youss2f",
    
    [Parameter(Mandatory=$false)]
    [string]$GitHubRepository = "fraudguard-360",
    
    [Parameter(Mandatory=$false)]
    [int]$MaxPages = 20,
    
    [Parameter(Mandatory=$false)]
    [switch]$WhatIf,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Configuration
$uriBase = "https://api.github.com"
$baseHeader = @{
    "Authorization" = "token $GitHubToken"
    "Content-Type" = "application/json"
    "Accept" = "application/vnd.github.v3+json"
}

Write-Host "GitHub Actions Bulk Cleanup Script" -ForegroundColor Green
Write-Host "Repository: $GitHubUser/$GitHubRepository" -ForegroundColor Yellow
Write-Host "Max Pages: $MaxPages" -ForegroundColor Yellow
Write-Host "WhatIf Mode: $($WhatIf.IsPresent)" -ForegroundColor Yellow
Write-Host ""

try {
    # Collect all failed runs with pagination
    Write-Host "Fetching all failed workflow runs..." -ForegroundColor Cyan
    $allFailedRuns = @()
    $page = 1
    $perPage = 100
    $totalRuns = 0
    
    do {
        Write-Host "  Fetching page $page..." -ForegroundColor Gray
        
        $runsPage = Invoke-RestMethod -Uri "$uriBase/repos/$GitHubUser/$GitHubRepository/actions/runs?page=$page&per_page=$perPage" -Headers $baseHeader -Method Get
        
        $failedOnPage = $runsPage.workflow_runs | Where-Object { 
            $_.conclusion -eq "failure" -or $_.conclusion -eq "cancelled" -or $_.conclusion -eq "timed_out" 
        }
        
        $allFailedRuns += $failedOnPage
        $totalRuns += $runsPage.workflow_runs.Count
        
        Write-Host "    Found $($failedOnPage.Count) failed runs on page $page (Total runs on page: $($runsPage.workflow_runs.Count))" -ForegroundColor Yellow
        
        $page++
        Start-Sleep -Milliseconds 100  # Rate limiting
        
    } while ($runsPage.workflow_runs.Count -eq $perPage -and $page -le $MaxPages)
    
    Write-Host "`nSummary:" -ForegroundColor Green
    Write-Host "Total runs scanned: $totalRuns" -ForegroundColor Yellow
    Write-Host "Failed/Cancelled/Timed-out runs found: $($allFailedRuns.Count)" -ForegroundColor Red
    
    if ($allFailedRuns.Count -eq 0) {
        Write-Host "No failed workflow runs to clean up!" -ForegroundColor Green
        return
    }
    
    # Group by status for summary
    $statusGroups = $allFailedRuns | Group-Object conclusion
    Write-Host "`nBreakdown by status:" -ForegroundColor Yellow
    foreach ($group in $statusGroups) {
        Write-Host "  $($group.Name): $($group.Count) runs" -ForegroundColor Gray
    }
    
    if ($WhatIf) {
        Write-Host "`n[WhatIf] Would delete $($allFailedRuns.Count) failed workflow runs" -ForegroundColor Magenta
        return
    }
    
    # Confirm deletion unless forced
    if (-not $Force) {
        Write-Host "`nWARNING: This will delete $($allFailedRuns.Count) workflow runs!" -ForegroundColor Red
        $confirmation = Read-Host "Do you want to proceed with bulk deletion? Type 'DELETE' to confirm"
        if ($confirmation -ne 'DELETE') {
            Write-Host "Operation cancelled." -ForegroundColor Yellow
            return
        }
    }
    
    # Delete failed runs in batches
    Write-Host "`nStarting bulk deletion..." -ForegroundColor Red
    $deletedCount = 0
    $errorCount = 0
    $batchSize = 10
    $batches = [Math]::Ceiling($allFailedRuns.Count / $batchSize)
    
    for ($i = 0; $i -lt $allFailedRuns.Count; $i += $batchSize) {
        $currentBatch = [Math]::Floor($i / $batchSize) + 1
        $batch = $allFailedRuns[$i..([Math]::Min($i + $batchSize - 1, $allFailedRuns.Count - 1))]
        
        Write-Host "Processing batch $currentBatch of $batches ($($batch.Count) runs)..." -ForegroundColor Cyan
        
        foreach ($run in $batch) {
            try {
                $runsDeleteParam = @{
                    Uri     = "$uriBase/repos/$GitHubUser/$GitHubRepository/actions/runs/$($run.id)"
                    Method  = "Delete"
                    Headers = $baseHeader
                }
                
                Write-Host "  Deleting run ID: $($run.id) ($($run.name)) - $($run.conclusion)" -ForegroundColor Gray
                Invoke-RestMethod @runsDeleteParam
                $deletedCount++
                
            }
            catch {
                Write-Warning "Failed to delete run ID $($run.id): $($_.Exception.Message)"
                $errorCount++
            }
        }
        
        # Progress update
        $progress = [Math]::Round(($deletedCount + $errorCount) / $allFailedRuns.Count * 100, 1)
        Write-Host "Progress: $progress% ($deletedCount deleted, $errorCount errors)" -ForegroundColor Yellow
        
        # Rate limiting between batches
        if ($currentBatch -lt $batches) {
            Start-Sleep -Milliseconds 500
        }
    }
    
    Write-Host "`nBulk cleanup completed!" -ForegroundColor Green
    Write-Host "Successfully deleted: $deletedCount runs" -ForegroundColor Green
    if ($errorCount -gt 0) {
        Write-Host "Failed to delete: $errorCount runs" -ForegroundColor Red
    }
    
    # Final verification
    Write-Host "`nVerifying cleanup..." -ForegroundColor Cyan
    $verifyRuns = Invoke-RestMethod -Uri "$uriBase/repos/$GitHubUser/$GitHubRepository/actions/runs?per_page=100" -Headers $baseHeader -Method Get
    $remainingFailed = $verifyRuns.workflow_runs | Where-Object { 
        $_.conclusion -eq "failure" -or $_.conclusion -eq "cancelled" -or $_.conclusion -eq "timed_out" 
    }
    
    Write-Host "Remaining failed runs in first 100: $($remainingFailed.Count)" -ForegroundColor $(if ($remainingFailed.Count -eq 0) { "Green" } else { "Yellow" })
    
}
catch {
    Write-Error "Error occurred: $($_.Exception.Message)"
    exit 1
}