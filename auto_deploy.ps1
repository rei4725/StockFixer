# auto_deploy.ps1
# develop へのマージをトリガーにした自動デプロイ（health-check + ロールバック + 失敗 Issue）。
#
# 想定運用: Windows タスクスケジューラから数分間隔で起動する。
#   - origin/develop の HEAD が前回処理時から変化していなければ即終了（= マージ検知トリガー）
#   - 変化していれば: pull -> テスト -> version タグでビルド -> up -d -> health-check
#   - health-check 失敗時: 直前イメージへロールバックし deploy-failure Issue を起票/更新
#   - 成功時: open な deploy-failure Issue を自動クローズ（壊れてた→直った の状態管理）
#
# Discord 通知は使わない（失敗は GitHub Issue で追跡、成功はログのみ）。

# 注意: $ErrorActionPreference は Stop にしない。
# git/docker など native コマンドが stderr に出力すると PS 5.1 が
# それを終了エラー扱いして誤検知するため。失敗判定は $LASTEXITCODE で行う。

$repoDir   = "C:\src\StockFixer"
$logDir    = Join-Path $repoDir "Logs"
$logFile   = Join-Path $logDir "auto_deploy.log"
$lockFile  = Join-Path $logDir ".auto_deploy.lock"
$stateFile = Join-Path $logDir "last_attempted_commit.txt"
$repo      = "rei4725/StockFixer"
$failLabel = "deploy-failure"
$lockMaxAgeMin = 40   # これより古いロックは stale とみなし上書き

function Write-Log($msg) {
    $ts   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts $msg"
    Write-Host $line
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    [System.IO.File]::AppendAllText($logFile, "$line`n", [System.Text.Encoding]::UTF8)
}

# 直近ログ末尾を Issue 本文用に取得
function Get-LogTail([int]$n = 40) {
    if (Test-Path $logFile) {
        return (Get-Content $logFile -Tail $n -Encoding UTF8) -join "`n"
    }
    return "(ログなし)"
}

# 失敗 Issue を起票 or 既存にコメント（冪等）
function Submit-FailureIssue($title, $body) {
    try {
        # ラベルが無ければ作成（既存ならエラーを握り潰す）
        try { gh label create $failLabel --repo $repo --color "B60205" --description "自動デプロイ失敗（要トリアージ）" 2>$null | Out-Null } catch {}
        $open = (gh issue list --repo $repo --label $failLabel --state open --json number --jq ".[].number" 2>$null)
        if ($open) {
            $num = ($open -split "\r?\n")[0]
            gh issue comment $num --repo $repo --body $body 2>$null | Out-Null
            Write-Log "[issue] 既存 deploy-failure #$num にコメント追記"
        } else {
            gh issue create --repo $repo --title $title --label $failLabel --body $body 2>$null | Out-Null
            Write-Log "[issue] deploy-failure Issue を新規起票"
        }
    } catch {
        Write-Log "[issue] Issue 起票/更新に失敗: $_"
    }
}

# 成功時: open な deploy-failure を自動クローズ
function Resolve-FailureIssues($deployedVersion) {
    try {
        $open = (gh issue list --repo $repo --label $failLabel --state open --json number --jq ".[].number" 2>$null)
        foreach ($num in ($open -split "\r?\n" | Where-Object { $_ })) {
            gh issue close $num --repo $repo --comment "✅ デプロイが成功（v$deployedVersion）したため自動クローズします。" 2>$null | Out-Null
            Write-Log "[issue] deploy-failure #$num を自動クローズ"
        }
    } catch {
        Write-Log "[issue] Issue クローズに失敗: $_"
    }
}

# health-check: healthy になるまでポーリング
function Wait-Healthy([int]$timeoutSec = 180) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        $status = (docker inspect --format "{{.State.Health.Status}}" stockfixer 2>$null)
        if ($status -eq "healthy") { return $true }
        if ($status -eq "unhealthy") { return $false }
        Start-Sleep -Seconds 5
    }
    return $false
}

# pytest ゲート（失敗で $true を返す）
function Invoke-TestGate($pythonExe, $label, [string[]]$pytestArgs) {
    Write-Log "[$label] python -m pytest $($pytestArgs -join ' ')"
    $failed = $false
    Push-Location (Join-Path $repoDir "python")
    try {
        & $pythonExe -m pytest @pytestArgs 2>&1 | Out-String -Stream | ForEach-Object {
            Write-Log "  [$label] $_"
            if ($_ -match "\d+ failed") { $script:failed = $true }
        }
        # exit=5 (no tests collected) は許容
        if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 5) { $failed = $true }
    } finally {
        Pop-Location
    }
    return $failed
}

# ---- ロック（多重起動防止） ----
if (Test-Path $lockFile) {
    $age = (New-TimeSpan -Start (Get-Item $lockFile).LastWriteTime -End (Get-Date)).TotalMinutes
    if ($age -lt $lockMaxAgeMin) {
        Write-Host "別のデプロイが実行中（lock age=$([int]$age)min）。終了します。"
        exit 0
    }
}
Set-Content -Path $lockFile -Value "$PID $(Get-Date -Format o)" -Encoding UTF8

$rollbackVersion = $null
try {
    Set-Location $repoDir

    # ---- develop の変化検知（マージ検知トリガー） ----
    git fetch origin develop 2>$null
    if ($LASTEXITCODE -ne 0) { throw "GIT: git fetch が失敗しました (exit=$LASTEXITCODE)" }
    $remoteSha = (git rev-parse origin/develop 2>$null).Trim()
    if (-not $remoteSha) { throw "GIT: origin/develop の SHA 取得に失敗" }
    $lastSha = if (Test-Path $stateFile) { (Get-Content $stateFile -Raw).Trim() } else { "" }
    if ($remoteSha -eq $lastSha) {
        # 変化なし=デプロイ不要。静かに終了。
        exit 0
    }

    Write-Log "=== 自動デプロイ開始 (develop $($remoteSha.Substring(0,7))) ==="

    $pythonExe = Join-Path $repoDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) { throw "Python 実行ファイルが見つかりません: $pythonExe" }

    # ---- ロールバック先 = 現在稼働中のイメージタグ ----
    $runningImage = (docker inspect --format "{{.Config.Image}}" stockfixer 2>$null)
    if ($runningImage -match "stockfixer:(.+)$") { $rollbackVersion = $Matches[1] }
    Write-Log "現在稼働イメージ: $runningImage (rollback=$rollbackVersion)"

    # ---- pull & 依存同期 ----
    Write-Log "[git] git pull origin develop"
    git pull origin develop 2>&1 | ForEach-Object { Write-Log "  [git] $_" }
    if ($LASTEXITCODE -ne 0) { throw "git pull が失敗しました (exit=$LASTEXITCODE)" }

    Push-Location (Join-Path $repoDir "python")
    try {
        & $pythonExe -m pip install -r requirements.txt -r requirements-dev.txt --quiet --prefer-binary 2>&1 | Out-String -Stream | ForEach-Object { Write-Log "  [pip] $_" }
        if ($LASTEXITCODE -ne 0) { throw "pip install が失敗しました (exit=$LASTEXITCODE)" }
    } finally { Pop-Location }

    # ---- デプロイ前テストゲート ----
    $stamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $smokeFailed = Invoke-TestGate $pythonExe "smoke" @(
        "tests/unit/test_data_pipeline.py","tests/unit/test_db_market_data.py",
        "tests/unit/test_db_prediction.py","tests/unit/test_db_stock_features.py",
        "tests/unit/test_shadow_mode.py","tests/unit/test_lgbm_xgb_models.py",
        "-q","--tb=short","--basetemp=.pytest_tmp_runs\smoke_$stamp")
    if ($smokeFailed) { throw "TEST_GATE: Smoke Test 失敗" }

    $unitFailed = Invoke-TestGate $pythonExe "unit" @("tests/unit","-q","--basetemp=.pytest_tmp_runs\unit_$stamp")
    if ($unitFailed) { throw "TEST_GATE: Unit Test 失敗" }

    $e2eFailed = Invoke-TestGate $pythonExe "e2e" @("tests/e2e","--timeout=300","-m","not slow","-q","--basetemp=.pytest_tmp_runs\e2e_$stamp")
    if ($e2eFailed) { throw "TEST_GATE: E2E Test 失敗" }

    # ---- ビルド & デプロイ ----
    $env:VERSION    = (Get-Content (Join-Path (Join-Path $repoDir "python") "VERSION")).Trim()
    $env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
    $env:GIT_COMMIT = (git rev-parse --short HEAD)
    Write-Log "[docker] build & up -d  VERSION=$($env:VERSION) COMMIT=$($env:GIT_COMMIT)"
    docker-compose up -d --build 2>&1 | ForEach-Object { Write-Log "  [docker] $_" }
    if ($LASTEXITCODE -ne 0) { throw "DOCKER: docker-compose up が失敗しました (exit=$LASTEXITCODE)" }

    # ---- health-check ----
    Write-Log "[health] healthy を待機中..."
    if (Wait-Healthy -timeoutSec 180) {
        Write-Log "[health] healthy 確認 ✅"
        Set-Content -Path $stateFile -Value $remoteSha -Encoding UTF8
        Resolve-FailureIssues $env:VERSION
        Write-Log "=== 自動デプロイ完了 (v$($env:VERSION)) ==="
        exit 0
    }

    throw "HEALTH: 新コンテナが healthy になりませんでした"

} catch {
    $errMsg = "$_"
    Write-Log "FAILED: $errMsg"

    # ---- ロールバック（既にコンテナを差し替えている場合のみ） ----
    $rolledBack = "未実施"
    if ($rollbackVersion -and ($errMsg -match "^HEALTH" -or $errMsg -match "^DOCKER")) {
        try {
            Write-Log "[rollback] v$rollbackVersion へロールバックします"
            $env:VERSION = $rollbackVersion
            docker-compose up -d 2>&1 | ForEach-Object { Write-Log "  [rollback] $_" }
            if (Wait-Healthy -timeoutSec 120) {
                $rolledBack = "成功 (v$rollbackVersion へ復帰・healthy)"
                Write-Log "[rollback] $rolledBack"
            } else {
                $rolledBack = "失敗 (v$rollbackVersion が healthy にならず・要手動対応)"
                Write-Log "[rollback] $rolledBack"
            }
        } catch {
            $rolledBack = "例外で失敗: $_"
            Write-Log "[rollback] $rolledBack"
        }
    }

    # ---- 失敗 Issue（冪等） ----
    # テストゲート失敗はコンテナ未変更なのでロールバック不要
    $stage = if ($errMsg -match "^TEST_GATE") { "デプロイ前テスト" } elseif ($errMsg -match "^HEALTH") { "health-check" } elseif ($errMsg -match "^DOCKER") { "docker ビルド/起動" } else { "デプロイ" }
    $shaShort = $remoteSha.Substring(0, 7)
    $nowStr = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $fence = ([char]96).ToString() * 3   # 三連バックティック（コードフェンス）
    $bodyLines = @(
        "## 自動デプロイ失敗",
        "",
        "| 項目 | 値 |",
        "|---|---|",
        "| 発生時刻 | $nowStr JST |",
        "| 失敗ステージ | $stage |",
        "| 対象 develop | $shaShort |",
        "| ロールバック | $rolledBack |",
        "| エラー | $errMsg |",
        "",
        "> 注意: デプロイ失敗の多くは環境・インフラ起因（DB ロック / Docker / ネット等）で、コードバグとは限りません。まずトリアージし、コード起因と判明したら auto-ok ラベルを付けて IssueAgent に回してください。",
        "",
        "### ログ末尾",
        $fence,
        (Get-LogTail 40),
        $fence
    )
    $body = $bodyLines -join "`n"
    Submit-FailureIssue "🚨 自動デプロイ失敗 (develop $shaShort)" $body

    # この commit は試行済みとして記録（develop が次に進むまで無限リトライしない）
    Set-Content -Path $stateFile -Value $remoteSha -Encoding UTF8
    exit 1
} finally {
    if (Test-Path $lockFile) { Remove-Item $lockFile -Force -ErrorAction SilentlyContinue }
}
