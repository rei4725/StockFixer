# register_auto_deploy_task.ps1
# auto_deploy.ps1 を 10 分間隔で実行するタスクスケジューラ登録ヘルパー。
# 管理者 PowerShell で実行すること。
#
# 解除: Unregister-ScheduledTask -TaskName "StockFixer Auto Deploy" -Confirm:$false

$taskName = "StockFixer Auto Deploy"
$script   = "C:\src\StockFixer\auto_deploy.ps1"

if (-not (Test-Path $script)) { throw "スクリプトが見つかりません: $script" }

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""

# 起動直後から 10 分間隔で無期限に繰り返す
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 10)

# 多重起動はスクリプト側ロックでも防ぐが、タスク側でも新規起動を抑止
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -RunLevel Highest -Force

Write-Host "登録完了: '$taskName'（10 分間隔）"
Write-Host "注意: 既存の 'StockFixer Weekly Redeploy' は重複デプロイ回避のため無効化を検討してください。"
