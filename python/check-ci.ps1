# PR 前に CI 相当のチェックをローカルで実行するスクリプト
# 使い方: cd python; .\check-ci.ps1
# 引数の設定は pyproject.toml / .flake8 が正本。このスクリプトは呼ぶだけ。
param([switch]$StopOnFail)

$ErrorActionPreference = "Continue"
$failed = @()

function Run-Step {
    param([string]$Name, [scriptblock]$Cmd)
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name" -ForegroundColor Red
        $script:failed += $Name
        if ($StopOnFail) { exit 1 }
    } else {
        Write-Host "OK: $Name" -ForegroundColor Green
    }
}

# 1. フォーマット（設定は pyproject.toml の [tool.black] / [tool.isort]）
Run-Step "black (check)" { python -m black . --check }
Run-Step "isort (check)" { python -m isort . --check }

# 2. Lint（設定は .flake8 / pyproject.toml の [tool.mypy]）
Run-Step "flake8"              { python -m flake8 . }
Run-Step "mypy"                { python -m mypy src/ }
Run-Step "pylint (errors only)" { python -m pylint src/ --rcfile=.pylintrc }

# 3. アーキテクチャ（設定は .importlinter）
Run-Step "import-linter" { lint-imports }

# 3.5 ファイル行数ゲート（肥大化防止 / Issue #497 フォローアップ）
Run-Step "file-size-check" { python scripts/check_file_size.py }

# 3.6 PostgreSQL起動確認（テストは _isolate_db 経由で実DB接続が必要）
Write-Host "PostgreSQL起動確認..."
docker compose up -d postgres
docker compose exec postgres pg_isready -U stockfixer -d stockfixer
if ($LASTEXITCODE -ne 0) {
    Write-Error "PostgreSQLが起動していません。docker compose up -d postgres を確認してください。"
    exit 1
}
$env:DATABASE_URL = "postgresql://stockfixer:stockfixer_dev@localhost:5432/stockfixer"

# 4. テスト＋カバレッジゲート（設定は pytest.ini）
# LightGBM/XGBoost を実学習させる回帰テスト(#615)は、xdist の並列ワーカー間で
# ネイティブ拡張の内部処理が競合し、CPUが限られた環境ではワーカー数を絞っても・
# n_jobs=1に固定しても flaky 化することを確認したため、このファイルだけ並列実行から
# 除外し直列で実行する（カバレッジは --cov-append で合算）。
Run-Step "unit tests (cov ≥80%)" {
    python -m pytest tests/unit/ -n 2 -v `
        --ignore=tests/unit/test_predict_unified_lightgbm_alignment.py `
        --cov=src --cov-branch --cov-report= --cov-fail-under=0
    $rc1 = $LASTEXITCODE
    python -m pytest tests/unit/test_predict_unified_lightgbm_alignment.py -v `
        --cov=src --cov-branch --cov-append --cov-report=term-missing --cov-fail-under=80
    if ($rc1 -ne 0) { $global:LASTEXITCODE = $rc1 }
}

# 5. SAST（bandit が入っている場合のみ）
if (Get-Command bandit -ErrorAction SilentlyContinue) {
    Run-Step "bandit (SAST)" { bandit -r src/ -ll --exclude src/_deprecated -q }
} else {
    Write-Host "`n--- bandit: スキップ (pip install bandit で有効化) ---" -ForegroundColor Yellow
}

# 6. 脆弱性スキャン（pip-audit が入っている場合のみ）
if (Get-Command pip-audit -ErrorAction SilentlyContinue) {
    Run-Step "pip-audit (requirements.txt)"     { pip-audit -r requirements.txt }
    Run-Step "pip-audit (requirements-dev.txt)" { pip-audit -r requirements-dev.txt }
} else {
    Write-Host "`n--- pip-audit: スキップ (pip install pip-audit で有効化) ---" -ForegroundColor Yellow
}

# 結果サマリー
Write-Host "`n==============================" -ForegroundColor Cyan
if ($failed.Count -eq 0) {
    Write-Host "All checks passed. Safe to open PR." -ForegroundColor Green
} else {
    Write-Host "Failed steps ($($failed.Count)):" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
