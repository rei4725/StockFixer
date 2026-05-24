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

# 4. テスト＋カバレッジゲート（設定は pytest.ini）
Run-Step "unit tests (cov ≥80%)" {
    python -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
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
