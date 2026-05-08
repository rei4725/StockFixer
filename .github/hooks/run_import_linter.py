#!/usr/bin/env python3
"""pre-commit から import-linter を python/ 配下で実行するラッパー。"""
import subprocess
import sys
from pathlib import Path

python_dir = Path(__file__).resolve().parents[2] / "python"
result = subprocess.run([sys.executable, "-m", "importlinter"], cwd=str(python_dir))
sys.exit(result.returncode)
