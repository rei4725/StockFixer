# register_task.ps1
# タスクスケジューラへの登録スクリプト（管理者権限で実行）
# 実行方法: このファイルを右クリック → "PowerShellで実行" または
#           管理者PowerShellで: .\register_task.ps1

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
    -Description "StockFixer Dockerコンテナの週次再ビルド・再起動（毎週土曜 04:00）" `
    -Force

Write-Host "登録完了。タスク名: 'StockFixer Weekly Redeploy'" -ForegroundColor Green
Write-Host "次回実行: 毎週土曜 04:00" -ForegroundColor Cyan
Write-Host ""
Write-Host "即時テスト実行:"
Write-Host "  Start-ScheduledTask -TaskName 'StockFixer Weekly Redeploy'" -ForegroundColor Yellow
