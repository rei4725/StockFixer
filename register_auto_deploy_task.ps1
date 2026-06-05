# register_auto_deploy_task.ps1
# auto_deploy.ps1 を 10 分間隔で実行するタスクスケジューラ登録ヘルパー。
# 管理者 PowerShell で実行すること。
#
# ウィンドウを出さず裏で実行するため「ユーザーがログオンしているかに関わらず実行」
# (LogonType Password) で登録する。これによりセッション0で動きコンソール窓が一切出ない。
# ※ パスワードを保存しない S4U モードはネットワーク不可で git/gh が失敗するため使わない。
#
# 解除: Unregister-ScheduledTask -TaskName "StockFixer Auto Deploy" -Confirm:$false

$taskName = "StockFixer Auto Deploy"
$script   = "C:\src\StockFixer\auto_deploy.ps1"
$user     = "$env:USERDOMAIN\$env:USERNAME"

if (-not (Test-Path $script)) { throw "スクリプトが見つかりません: $script" }

# 念のため -WindowStyle Hidden も付与（Password モードでは元々窓は出ないが多重防御）
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""

# 起動直後から 10 分間隔で無期限に繰り返す
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 10)

# 多重起動はスクリプト側ロックでも防ぐが、タスク側でも新規起動を抑止。
# -Hidden でタスク一覧上も非表示寄りにする。
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 1) -Hidden

# Windows パスワードを取得（資格情報マネージャに暗号化保存される）
$cred = Get-Credential -UserName $user -Message "自動デプロイタスク実行用に Windows パスワードを入力してください（裏で実行するために必要）"
$plainPwd = $cred.GetNetworkCredential().Password

# LogonType Password = ログオン状態に関わらずバックグラウンド実行（ウィンドウなし・ネットワーク可）
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -User $user -Password $plainPwd -RunLevel Highest -Force

Write-Host "登録完了: '$taskName'（10 分間隔・ウィンドウ非表示のバックグラウンド実行）"
Write-Host "注意: 既存の 'StockFixer Weekly Redeploy' は重複デプロイ回避のため無効化済み/無効化を推奨。"
