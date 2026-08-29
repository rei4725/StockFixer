#!/usr/bin/env bash
set -euo pipefail
LAYER="${1:-unit}"

case "$LAYER" in
  unit)        python -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80 ;;
  integration) python -m pytest tests/integration/ -v --timeout=120 ;;
  e2e)         python -m pytest tests/e2e/ -v --timeout=60 -m "not slow" ;;
  e2e-slow)    python -m pytest tests/e2e/ -v --timeout=300 -m "slow" ;;
  all)         python -m pytest tests/unit/ tests/integration/ tests/e2e/ -v -m "not slow" ;;
  *) echo "Usage: $0 {unit|integration|e2e|e2e-slow|all}" >&2; exit 1 ;;
esac
