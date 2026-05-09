# weekly_redeploy.ps1
# Windowsタスクスケジューラから毎週月曜 02:00 に呼び出される
# 処理: git pull -> UnitTest -> docker-compose up --build -> 結果ログ

$repoDir = "C:\src\StockFixer"
$logDir  = Join-Path $repoDir "Logs"
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

    # pytest の作業ディレクトリ衝突を避ける（WinError 5 対策）
    $smokeBaseTemp = ".pytest_tmp_runs\\smoke_$([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())"
    $unitBaseTemp  = ".pytest_tmp_runs\\unit_$([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())"
    $e2eBaseTemp   = ".pytest_tmp_runs\\e2e_$([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds())"

    # --- git pull ---
    Write-Log "[git] git pull origin develop"
    git pull origin develop 2>&1 | ForEach-Object { Write-Log "  [git] $_" }
    Write-Log "[git] 完了 (exit=$LASTEXITCODE)"
    if ($LASTEXITCODE -ne 0) { throw "git pull が失敗しました (exit=$LASTEXITCODE)" }

    # --- 依存パッケージ同期（git pull 後に追加されたパッケージを確実にインストール） ---
    Write-Log "[pip] pip install -r requirements.txt -r requirements-dev.txt"
    Push-Location (Join-Path $repoDir "python")
    try {
        & $pythonExe -m pip install -r requirements.txt -r requirements-dev.txt --quiet 2>&1 | Out-String -Stream | ForEach-Object { Write-Log "  [pip] $_" }
        if ($LASTEXITCODE -ne 0) { throw "pip install が失敗しました (exit=$LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
    Write-Log "[pip] 完了 (exit=$LASTEXITCODE)"

    # --- Smoke Test（失敗時は全量テスト前に中断） ---
    Write-Log "[smoke] python -m pytest tests/unit/test_data_pipeline.py tests/unit/test_db_market_data.py tests/unit/test_db_prediction.py tests/unit/test_db_stock_features.py tests/unit/test_shadow_mode.py tests/unit/test_lgbm_xgb_models.py -q --tb=short --basetemp=$smokeBaseTemp"
    $smokeHasFailed = $false
    Push-Location (Join-Path $repoDir "python")
    try {
        & $pythonExe -m pytest tests/unit/test_data_pipeline.py tests/unit/test_db_market_data.py tests/unit/test_db_prediction.py tests/unit/test_db_stock_features.py tests/unit/test_shadow_mode.py tests/unit/test_lgbm_xgb_models.py -q --tb=short --basetemp=$smokeBaseTemp 2>&1 | Out-String -Stream | ForEach-Object {
            Write-Log "  [smoke] $_"
            if ($_ -match "\d+ failed") { $script:smokeHasFailed = $true }
        }
        if ($LASTEXITCODE -ne 0) { $smokeHasFailed = $true }
    } finally {
        Pop-Location
    }
    Write-Log "[smoke] 完了 (exit=$LASTEXITCODE)"
    if ($smokeHasFailed) { throw "Smoke Test が失敗しました。デプロイを中断します (exit=$LASTEXITCODE)" }

    # --- UnitTest（失敗時はデプロイ中断） ---
    Write-Log "[test] python -m pytest tests/unit -v --basetemp=$unitBaseTemp"
    $testHasFailed = $false
    Push-Location (Join-Path $repoDir "python")
    try {
        # Out-String -Stream で ErrorRecord を含む全オブジェクトを文字列化してから処理
        & $pythonExe -m pytest tests/unit -v --basetemp=$unitBaseTemp 2>&1 | Out-String -Stream | ForEach-Object {
            Write-Log "  [test] $_"
            # pytest のサマリー行パターン: "1 failed, 3 passed" または "5 failed"
            if ($_ -match "\d+ failed") { $script:testHasFailed = $true }
        }
        # 収集エラー(exit=2)や内部エラー(exit=3)も確実に検知
        if ($LASTEXITCODE -ne 0) { $testHasFailed = $true }
    } finally {
        Pop-Location
    }
    Write-Log "[test] 完了 (exit=$LASTEXITCODE)"
    if ($testHasFailed) { throw "UnitTest が失敗しました。デプロイを中断します (exit=$LASTEXITCODE)" }

    # --- E2E テスト（失敗時はデプロイ中断） ---
    Write-Log "[e2e] python -m pytest tests/e2e -v --timeout=300 -m 'not slow' --basetemp=$e2eBaseTemp"
    $e2eHasFailed = $false
    Push-Location (Join-Path $repoDir "python")
    try {
        & $pythonExe -m pytest tests/e2e -v --timeout=300 -m "not slow" --basetemp=$e2eBaseTemp 2>&1 | Out-String -Stream | ForEach-Object {
            Write-Log "  [e2e] $_"
            if ($_ -match "\d+ failed") { $script:e2eHasFailed = $true }
        }
        # exit=1(failed) / exit=2(collection error) / exit=3(internal error) を検知
        # exit=5(no tests collected) は問題なしとみなす
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 5) { $e2eHasFailed = $true }
    } finally {
        Pop-Location
    }
    Write-Log "[e2e] 完了 (exit=$LASTEXITCODE)"
    if ($e2eHasFailed) { throw "E2E テストが失敗しました。デプロイを中断します (exit=$LASTEXITCODE)" }

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
