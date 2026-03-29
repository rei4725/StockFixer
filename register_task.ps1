# register_task.ps1
# Register StockFixer weekly redeploy to Windows Task Scheduler (run as Administrator)
# Usage: Right-click -> "Run with PowerShell"  OR  .\register_task.ps1 (in admin PS)

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File C:\src\StockFixer\weekly_redeploy.ps1"

$trigger = New-ScheduledTaskTrigger `
    -Weekly -DaysOfWeek Saturday -At "04:00"

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName   "StockFixer Weekly Redeploy" `
    -Action     $action `
    -Trigger    $trigger `
    -Settings   $settings `
    -RunLevel   Highest `
    -Description "StockFixer Docker weekly rebuild and restart (every Saturday 04:00)" `
    -Force

Write-Host '[OK] Registered: StockFixer Weekly Redeploy' -ForegroundColor Green
Write-Host '     Schedule : Every Saturday 04:00' -ForegroundColor Cyan
Write-Host ''
Write-Host 'Run manually:'
Write-Host '  Start-ScheduledTask -TaskName "StockFixer Weekly Redeploy"' -ForegroundColor Yellow
