# docker_watchdog.ps1
# Docker エンジンが停止していたら起動し直すウォッチドッグ。
#
# 想定運用: auto_deploy.ps1 の冒頭から呼ばれる（= タスクスケジューラの 10 分間隔に相乗り）。
#   - docker info が通れば何もせず終了（ログも書かない = 正常時のログ汚染を避ける）
#   - 通らなければ `docker desktop start` でエンジンを起動し、復旧可否をログに残す
#
# コンテナ自体は docker-compose.yml の restart: always に任せる（エンジン復帰で自動的に戻る）。
# 通知はしない（auto_deploy.ps1 と同じく「成功はログのみ」方針）。
#
# 単体実行: powershell -ExecutionPolicy Bypass -File C:\src\StockFixer\docker_watchdog.ps1
#
# 注意: $ErrorActionPreference は Stop にしない。
# docker など native コマンドが stderr に出力すると PS 5.1 が
# それを終了エラー扱いして誤検知するため。失敗判定は $LASTEXITCODE で行う。

param(
    # docker desktop start の同期待ち上限（秒）
    [int]$TimeoutSec = 300
)

# native コマンド出力を UTF-8 として取り込む（docker の ✓ 等が化けるのを防ぐ）。
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$repoDir   = "C:\src\StockFixer"
$logDir    = Join-Path $repoDir "Logs"
$logFile   = Join-Path $logDir "docker_watchdog.log"
$lockFile  = Join-Path $logDir ".docker_watchdog.lock"
$lockMaxAgeMin = 30   # これより古いロックは stale とみなし上書き

# エンジン起動後、docker info が通るまでの保険ポーリング上限（秒）。
# docker desktop start は同期待ちするので通常は 0 回で抜ける。
$settleSec = 30

function Write-Log($msg) {
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $msg"
    Write-Host $line
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    [System.IO.File]::AppendAllText($logFile, "$line`n", [System.Text.Encoding]::UTF8)
}

# エンジンが応答するか（docker info の終了コードで判定）
function Test-DockerUp {
    docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

# ---- (1) 判定: 生きていれば即終了 ----
if (Test-DockerUp) { exit 0 }

# ---- ロック（多重起動防止） ----
if (Test-Path $lockFile) {
    $age = (New-TimeSpan -Start (Get-Item $lockFile).LastWriteTime -End (Get-Date)).TotalMinutes
    if ($age -lt $lockMaxAgeMin) {
        Write-Host "別のウォッチドッグが実行中（lock age=$([int]$age)min）。終了します。"
        exit 0
    }
}
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
Set-Content -Path $lockFile -Value "$PID $(Get-Date -Format o)" -Encoding UTF8

try {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Log "=== Docker エンジン停止を検知。起動を試みます ==="

    # ---- (2) 起動 ----
    docker desktop start --timeout $TimeoutSec 2>&1 | ForEach-Object { Write-Log "  [start] $_" }
    $startExit = $LASTEXITCODE

    # ---- (3) 確認（保険ポーリング） ----
    $deadline = (Get-Date).AddSeconds($settleSec)
    while (-not (Test-DockerUp) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
    }
    $sw.Stop()
    $elapsed = [int]$sw.Elapsed.TotalSeconds

    # ---- (4) 記録 ----
    if (Test-DockerUp) {
        Write-Log "[ok] 復旧成功（所要 ${elapsed} 秒 / docker desktop start exit=$startExit）"
        exit 0
    }

    Write-Log "[fail] 復旧せず（所要 ${elapsed} 秒 / docker desktop start exit=$startExit）"
    Write-Log "[fail] セッション 0（未ログオン）から GUI アプリを起動できなかった可能性があります。"
    exit 1

} catch {
    Write-Log "[fail] 例外により復旧できませんでした: $_"
    exit 1

} finally {
    if (Test-Path $lockFile) { Remove-Item $lockFile -Force -ErrorAction SilentlyContinue }
}
