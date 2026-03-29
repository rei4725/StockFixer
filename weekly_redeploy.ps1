# weekly_redeploy.ps1
# Windowsタスクスケジューラから毎週土曜 04:00 に呼び出される
# 処理: git pull -> UnitTest -> docker-compose up --build -> 結果ログ

$repoDir = "C:\src\StockFixer"
$logDir  = Join-Path $repoDir "python\logs"
$logFile = Join-Path $logDir "redeploy.log"

function Write-Log($msg) {
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $msg"
    Write-Host $line
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    [System.IO.File]::AppendAllText($logFile, "$line`n", [System.Text.Encoding]::UTF8)
}

Write-Log "=== 週次再デプロイ開始 ==="

try {
    Set-Location $repoDir

    $pythonExe = Join-Path $repoDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "Python 実行ファイルが見つかりません: $pythonExe"
    }

    # --- git pull ---
    Write-Log "[git] git pull origin feature/training"
    git pull origin feature/training 2>&1 | ForEach-Object { Write-Log "  [git] $_" }
    Write-Log "[git] 完了 (exit=$LASTEXITCODE)"
    if ($LASTEXITCODE -ne 0) { throw "git pull が失敗しました (exit=$LASTEXITCODE)" }

    # --- UnitTest（失敗時はデプロイ中断） ---
    Write-Log "[test] python -m pytest tests/unit -v"
    $testExitCode = 0
    Push-Location (Join-Path $repoDir "python")
    try {
        & $pythonExe -m pytest tests/unit -v 2>&1 | ForEach-Object { Write-Log "  [test] $_" }
        $testExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    Write-Log "[test] 完了 (exit=$testExitCode)"
    if ($testExitCode -ne 0) { throw "UnitTest が失敗しました。デプロイを中断します (exit=$testExitCode)" }

    # --- ビルド引数セット ---
    $env:VERSION    = (Get-Content (Join-Path $repoDir "VERSION")).Trim()
    $env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    $env:GIT_COMMIT = git rev-parse --short HEAD
    Write-Log "VERSION=$($env:VERSION)  GIT_COMMIT=$($env:GIT_COMMIT)  BUILD_DATE=$($env:BUILD_DATE)"

    # --- docker-compose ビルド&再起動 ---
    Write-Log "[docker] docker-compose up -d --build"
    docker-compose up -d --build 2>&1 | ForEach-Object { Write-Log "  [docker] $_" }
    Write-Log "[docker] 完了 (exit=$LASTEXITCODE)"
    if ($LASTEXITCODE -ne 0) { throw "docker-compose が失敗しました (exit=$LASTEXITCODE)" }

    # --- 起動確認 ---
    Start-Sleep -Seconds 5
    $status = docker ps --filter "name=stockfixer" --format "{{.Status}}"
    Write-Log "コンテナ状態: $status"

    Write-Log "=== 再デプロイ完了 ==="
    exit 0
} catch {
    Write-Log "FAILED: $_"
    exit 1
}
