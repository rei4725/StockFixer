#!/usr/bin/env bash
# PR 前に CI 相当のチェックをローカルで実行するスクリプト
# 使い方: cd python && bash check-ci.sh
# 引数の設定は pyproject.toml / .flake8 が正本。このスクリプトは呼ぶだけ。
# -s オプションで失敗時即停止
set -euo pipefail

STOP_ON_FAIL=0
while getopts "s" opt; do
  case $opt in s) STOP_ON_FAIL=1 ;; esac
done

FAILED=()

run_step() {
  local name="$1"; shift
  echo ""
  echo "=== $name ==="
  if "$@"; then
    echo "OK: $name"
  else
    echo "FAILED: $name"
    FAILED+=("$name")
    [ "$STOP_ON_FAIL" -eq 1 ] && exit 1
  fi
}

# 1. フォーマット（設定は pyproject.toml の [tool.black] / [tool.isort]）
run_step "black (check)" python -m black . --check
run_step "isort (check)" python -m isort . --check

# 2. Lint（設定は .flake8 / pyproject.toml の [tool.mypy]）
run_step "flake8"               python -m flake8 .
run_step "mypy"                 python -m mypy src/
run_step "pylint (errors only)" python -m pylint src/ --rcfile=.pylintrc

# 3. アーキテクチャ（設定は .importlinter）
run_step "import-linter" lint-imports

# 3.5 ファイル行数ゲート（肥大化防止 / Issue #497 フォローアップ）
run_step "file-size-check" python scripts/check_file_size.py

# 3.6 PostgreSQL起動確認（テストは _isolate_db 経由で実DB接続が必要）
echo "PostgreSQL起動確認..."
docker compose up -d postgres
docker compose exec postgres pg_isready -U stockfixer -d stockfixer || {
    echo "PostgreSQLが起動していません。docker compose up -d postgres を確認してください。" >&2
    exit 1
}
export DATABASE_URL="postgresql://stockfixer:stockfixer_dev@localhost:5432/stockfixer"

# 4. テスト＋カバレッジゲート（設定は pytest.ini）
# -n 2: CI(ubuntu-latest, 2vCPU)と揃える。LightGBM実学習を伴う回帰テスト(#615)が
# ワーカー過多だと内部リソース競合で flaky 化するため、-n auto は使わない。
run_step "unit tests (cov ≥80%)" \
  python -m pytest tests/unit/ -n 2 -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80

# 5. SAST
if command -v bandit &>/dev/null; then
  run_step "bandit (SAST)" bandit -r src/ -ll --exclude src/_deprecated -q
else
  echo "--- bandit: スキップ (pip install bandit で有効化) ---"
fi

# 6. 脆弱性スキャン
if command -v pip-audit &>/dev/null; then
  run_step "pip-audit (requirements.txt)"     pip-audit -r requirements.txt
  run_step "pip-audit (requirements-dev.txt)" pip-audit -r requirements-dev.txt
else
  echo "--- pip-audit: スキップ (pip install pip-audit で有効化) ---"
fi

# 結果サマリー
echo ""
echo "=============================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "All checks passed. Safe to open PR."
else
  echo "Failed steps (${#FAILED[@]}):"
  for s in "${FAILED[@]}"; do echo "  - $s"; done
  exit 1
fi
