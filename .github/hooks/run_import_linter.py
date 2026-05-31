#!/usr/bin/env python3
"""pre-commit から import-linter を python/ 配下で実行するラッパー。"""

import os
import subprocess
import sys
from pathlib import Path

python_dir = Path(__file__).resolve().parents[2] / "python"

env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
result = subprocess.run(
    [
        sys.executable,
        "-c",
        "from importlinter.cli import lint_imports_command; lint_imports_command()",
    ],
    cwd=str(python_dir),
    env=env,
)
sys.exit(result.returncode)
