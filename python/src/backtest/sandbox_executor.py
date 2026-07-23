"""Claude生成ルールコードを Docker サンドボックスで隔離実行するオーケストレーター。

AST静的検査（高速フィルタ、主たる安全境界ではない）→ Dockerサンドボックス実行
（--network none・読み取り専用マウント、主たる安全境界）の順で実行し、結果を
既存の FactoryEvaluation 互換の形で返す。生成コード自体はこのモジュール内では
一切execしない（execするのはサンドボックスコンテナ内の scripts/sandbox_evaluate_rule.py）。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.settings import (
    FACTORY_SANDBOX_CPU_LIMIT,
    FACTORY_SANDBOX_IMAGE,
    FACTORY_SANDBOX_MEMORY_LIMIT,
    FACTORY_SANDBOX_TIMEOUT_SECONDS,
)
from src.backtest.ast_safety_check import check_source_safety
from src.backtest.types import FactoryEvaluation, FactoryHypothesis
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SandboxRunResult:
    """サンドボックス実行1回分の結果。

    kind:
        "gate_evaluated" — 正常にバックテスト完了。evaluation を保持。
        "repairable"     — AST違反・実行時例外。repair_detail に修復用の情報。
                            ゲート判定結果はここに含まれない（別経路）。
        "infra_error"    — Dockerの起動失敗・タイムアウト等、コード起因でない。
    """

    kind: str
    evaluation: Optional[FactoryEvaluation] = None
    repair_detail: Optional[str] = None
    infra_detail: Optional[str] = None


def prepare_sandbox_data(
    data_by_symbol: dict[str, pd.DataFrame],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[str, str]:
    """バッチ全体で使い回す共有データディレクトリ/windowsファイルを1回だけ作る。

    呼び出し元が一晩のバッチにつき1回だけ呼び、返り値のパスを
    run_sandboxed_evaluation に渡す。呼び出し元は使用後にディレクトリを
    削除する責務を持つ（tempfile.TemporaryDirectory 等で管理）。
    """
    data_dir = tempfile.mkdtemp(prefix="factory_sandbox_data_")
    for symbol, df in data_by_symbol.items():
        df.to_parquet(os.path.join(data_dir, f"{symbol}.parquet"))

    windows_fd, windows_path = tempfile.mkstemp(prefix="factory_sandbox_windows_", suffix=".json")
    raw = [[w[0].isoformat(), w[1].isoformat()] for w in windows]
    with os.fdopen(windows_fd, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    return data_dir, windows_path


def _detect_self_image() -> str:
    """FACTORY_SANDBOX_IMAGE が未設定の場合、自コンテナのイメージ名を推測する。"""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.Image}}", os.environ.get("HOSTNAME", "")],
        capture_output=True,
        text=True,
        check=False,
    )
    image = result.stdout.strip()
    if not image:
        raise RuntimeError(
            "FACTORY_SANDBOX_IMAGE が未設定で、自コンテナのイメージ名も検出できませんでした"
        )
    return image


def run_sandboxed_evaluation(
    hypothesis: FactoryHypothesis,
    shared_data_dir: str,
    windows_file: str,
) -> SandboxRunResult:
    """1候補をAST検査 → Dockerサンドボックスで評価する。"""
    spec = hypothesis.rule_spec
    source_code = spec["source_code"]

    safety = check_source_safety(source_code)
    if not safety.passed:
        detail = "; ".join(f"{v.line}行目: {v.reason}" for v in safety.violations)
        return SandboxRunResult(kind="repairable", repair_detail=f"静的検査で拒否: {detail}")

    with tempfile.TemporaryDirectory(prefix="factory_sandbox_src_") as src_dir:
        source_path = os.path.join(src_dir, "candidate.py")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source_code)

        image = FACTORY_SANDBOX_IMAGE or _detect_self_image()
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/Logs",
            "--memory",
            FACTORY_SANDBOX_MEMORY_LIMIT,
            "--cpus",
            FACTORY_SANDBOX_CPU_LIMIT,
            "-e",
            "STOCKFIXER_SANDBOX=1",
            "-e",
            "PYTHONPATH=/app",
            "-v",
            f"{src_dir}:/sandbox/src:ro",
            "-v",
            f"{shared_data_dir}:/sandbox/data:ro",
            "-v",
            f"{windows_file}:/sandbox/windows.json:ro",
            image,
            "python",
            "scripts/sandbox_evaluate_rule.py",
            "--source-file",
            "/sandbox/src/candidate.py",
            "--class-name",
            spec["class_name"],
            "--rule-name",
            spec["rule_name"],
            "--description",
            spec["description"],
            "--market",
            hypothesis.market,
            "--lookback-years",
            str(hypothesis.lookback_years),
            "--data-dir",
            "/sandbox/data",
            "--windows-file",
            "/sandbox/windows.json",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=FACTORY_SANDBOX_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxRunResult(
                kind="infra_error", infra_detail="サンドボックス実行がタイムアウトしました"
            )

        if proc.returncode not in (0, 1):
            return SandboxRunResult(
                kind="infra_error",
                infra_detail=f"docker run 異常終了 (code={proc.returncode}): {proc.stderr[-2000:]}",
            )

        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return SandboxRunResult(
                kind="infra_error",
                infra_detail=f"サンドボックス出力の解析に失敗: {proc.stdout[-2000:]}",
            )

        if payload.get("status") == "error":
            return SandboxRunResult(
                kind="repairable",
                repair_detail=f"サンドボックス実行時エラー: {payload.get('traceback', '')[-2000:]}",
            )

        ev = payload["evaluation"]
        evaluation = FactoryEvaluation(
            hypothesis=hypothesis,
            sharpe_ratio=ev["sharpe_ratio"],
            sharpe_per_trade=ev["sharpe_per_trade"],
            win_rate=ev["win_rate"],
            num_trades=ev["num_trades"],
            max_drawdown=ev["max_drawdown"],
            total_return=ev["total_return"],
            window_returns=ev["window_returns"],
            n_symbols=ev["n_symbols"],
        )
        return SandboxRunResult(kind="gate_evaluated", evaluation=evaluation)
