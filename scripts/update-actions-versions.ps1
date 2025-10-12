# GitHub Actions Version Updater Script
# This script updates all workflow files to use the latest action versions

Write-Host "GitHub Actions Version Updater" -ForegroundColor Green
Write-Host "Updating workflow files to use latest action versions..." -ForegroundColor Yellow

# Define the action version mappings
$actionUpdates = @{
    "actions/setup-python@v4" = "actions/setup-python@v5"
    "actions/setup-python@v3" = "actions/setup-python@v5"
    "actions/setup-python@v2" = "actions/setup-python@v5"
    "actions/setup-node@v3" = "actions/setup-node@v4"
    "actions/setup-node@v2" = "actions/setup-node@v4"
    "actions/setup-java@v3" = "actions/setup-java@v4"
    "actions/setup-java@v2" = "actions/setup-java@v4"
    "actions/cache@v3" = "actions/cache@v4"
    "actions/cache@v2" = "actions/cache@v4"
    "actions/upload-artifact@v3" = "actions/upload-artifact@v4"
    "actions/upload-artifact@v2" = "actions/upload-artifact@v4"
    "actions/download-artifact@v3" = "actions/download-artifact@v4"
    "actions/download-artifact@v2" = "actions/download-artifact@v4"
    "actions/checkout@v3" = "actions/checkout@v4"
    "actions/checkout@v2" = "actions/checkout@v4"
}

# Get all workflow files
$workflowFiles = Get-ChildItem -Path ".\.github\workflows" -Filter "*.yml" -Recurse

Write-Host "Found $($workflowFiles.Count) workflow files to check" -ForegroundColor Cyan

$updatedFiles = 0
$totalReplacements = 0

foreach ($file in $workflowFiles) {
    $content = Get-Content $file.FullName -Raw
    $originalContent = $content
    $fileReplacements = 0
    
    # Apply all action updates
    foreach ($oldAction in $actionUpdates.Keys) {
        $newAction = $actionUpdates[$oldAction]
        if ($content -match [regex]::Escape($oldAction)) {
            $content = $content -replace [regex]::Escape($oldAction), $newAction
            $replacementCount = ([regex]::Matches($originalContent, [regex]::Escape($oldAction))).Count
            if ($replacementCount -gt 0) {
                $fileReplacements += $replacementCount
                Write-Host "  - Updated $oldAction -> $newAction ($replacementCount times)" -ForegroundColor Yellow
            }
        }
    }
    
    # Write the updated content if changes were made
    if ($originalContent -ne $content) {
        Set-Content -Path $file.FullName -Value $content -NoNewline
        Write-Host "Updated $($file.Name) - $fileReplacements replacements" -ForegroundColor Green
        $updatedFiles++
        $totalReplacements += $fileReplacements
    } else {
        Write-Host "No updates needed for $($file.Name)" -ForegroundColor Gray
    }
}

Write-Host "`nUpdate Summary:" -ForegroundColor Green
Write-Host "Files updated: $updatedFiles" -ForegroundColor Yellow
Write-Host "Total replacements: $totalReplacements" -ForegroundColor Yellow

if ($updatedFiles -gt 0) {
    Write-Host "`nAll workflow files have been updated to use the latest action versions!" -ForegroundColor Green
    Write-Host "Consider running 'git add . && git commit -m `"Update GitHub Actions to latest versions`"' to commit these changes." -ForegroundColor Cyan
} else {
    Write-Host "`nAll workflow files are already using the latest action versions!" -ForegroundColor Green
}