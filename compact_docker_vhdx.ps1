# compact_docker_vhdx.ps1
# Docker の仮想ディスク (docker_data.vhdx) を圧縮して C: の空きを取り戻す。
# 管理者 PowerShell で実行すること（Optimize-VHD が管理者権限を要する）。
#
# vhdx は中のデータを消しても自動では縮まない。docker_disk_check.ps1 が
# docker-disk-large の Issue を立てたら、これを実行する。
#   1. 自動デプロイ（中のウォッチドッグが Docker を起こし直す）と ai-company の起動を一時停止
#   2. Docker Desktop を止めて wsl --shutdown
#   3. Optimize-VHD -Mode Full で圧縮
#   4. 成否にかかわらず Docker を起動し直し、止めたタスクを戻す
#
# StockFixer のコンテナは数分止まる（2026-10-29 は 965GB → 78GB を 1.2 分で圧縮、全体で約 3 分）。
# 先に未使用のイメージ・ビルドキャッシュを prune しておくと、より縮む。
#
# 実行: powershell -ExecutionPolicy Bypass -File C:\src\StockFixer\compact_docker_vhdx.ps1

$ErrorActionPreference = 'Stop'
$repoDir = "C:\src\StockFixer"
$vhdx    = Join-Path $env:LOCALAPPDATA "Docker\wsl\disk\docker_data.vhdx"
$log     = Join-Path $repoDir "Logs\compact_docker_vhdx.log"
$tasks   = 'StockFixer Auto Deploy', 'ai-company - Run'
Start-Transcript -Path $log -Append | Out-Null

function Show-Size($label) {
    '{0}: vhdx {1:N1} GB / C: free {2:N1} GB' -f $label, ((Get-Item $vhdx).Length / 1GB), ((Get-PSDrive C).Free / 1GB)
}

$disabled = @()
try {
    Show-Size 'before'
    foreach ($t in $tasks) {
        if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
            Disable-ScheduledTask -TaskName $t | Out-Null
            $disabled += $t
            "disabled: $t"
        }
    }

    'stopping Docker Desktop...'
    try { docker desktop stop 2>&1 | Out-Host } catch { }
    Get-Process 'Docker Desktop', 'com.docker.backend' -ErrorAction SilentlyContinue | Stop-Process -Force
    wsl --shutdown
    Start-Sleep -Seconds 10

    'compacting (this can take a while)...'
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Optimize-VHD -Path $vhdx -Mode Full
    'compact done in {0:N1} min' -f $sw.Elapsed.TotalMinutes
    Show-Size 'after'
}
catch {
    "ERROR: $_"
}
finally {
    'starting Docker Desktop...'
    Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    for ($i = 0; $i -lt 60; $i++) {
        docker info *> $null
        if ($LASTEXITCODE -eq 0) { 'docker is up'; break }
        Start-Sleep -Seconds 5
    }
    foreach ($t in $disabled) { Enable-ScheduledTask -TaskName $t | Out-Null; "enabled: $t" }
    docker ps --format '{{.Names}}  {{.Status}}'
    Stop-Transcript | Out-Null
}
