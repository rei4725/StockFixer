"""Claude生成ルールコードを Docker サンドボックスで隔離実行するオーケストレーター。

AST静的検査（高速フィルタ、主たる安全境界ではない）→ Dockerサンドボックス実行
（--network none・読み取り専用マウント、主たる安全境界）の順で実行し、結果を
既存の FactoryEvaluation 互換の形で返す。生成コード自体はこのモジュール内では
一切execしない（execするのはサンドボックスコンテナ内の scripts/sandbox_evaluate_rule.py）。
"""

from __future__ import annotations

import json
import os
import posixpath
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.settings import (
    FACTORY_GATE_MIN_TRADES_PER_SYMBOL,
    FACTORY_SANDBOX_CPU_LIMIT,
    FACTORY_SANDBOX_IMAGE,
    FACTORY_SANDBOX_MEMORY_LIMIT,
    FACTORY_SANDBOX_SHARE_DIR,
    FACTORY_SANDBOX_SHARE_VOLUME,
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

    environment_broken=True は、サンドボックスが入力（候補ソース・株価データ）を
    読めなかったことを表す。候補を替えても直らないため、呼び出し元はその晩の生成を
    打ち切ること（修復を Claude に依頼しても API を空費するだけ。#757）。
    """

    kind: str
    evaluation: Optional[FactoryEvaluation] = None
    repair_detail: Optional[str] = None
    infra_detail: Optional[str] = None
    environment_broken: bool = False


# サンドボックス内で共有ボリュームをマウントする場所
_SANDBOX_SHARE_MOUNT = "/sandbox_share"


def _share_config() -> Optional[tuple[str, str]]:
    """共有ボリューム方式の設定 (本体内ディレクトリ, ボリューム名) を返す。未設定なら None。

    片方だけの設定は、一時ファイルをどこに置いてもサンドボックスから見えない誤設定のため
    例外にする（無言で bind mount 方式に落ちると #757 と同じ全滅が再発する）。
    """
    if FACTORY_SANDBOX_SHARE_DIR and FACTORY_SANDBOX_SHARE_VOLUME:
        return FACTORY_SANDBOX_SHARE_DIR, FACTORY_SANDBOX_SHARE_VOLUME
    if FACTORY_SANDBOX_SHARE_DIR or FACTORY_SANDBOX_SHARE_VOLUME:
        raise RuntimeError(
            "FACTORY_SANDBOX_SHARE_DIR と FACTORY_SANDBOX_SHARE_VOLUME は両方設定するか"
            "両方未設定にしてください（片方だけではサンドボックスに入力が渡りません）"
        )
    return None


def _workspace_dir() -> Optional[str]:
    """サンドボックスに渡す一時ファイルを作る親ディレクトリ（None なら OS 既定の一時領域）。"""
    config = _share_config()
    return config[0] if config else None


def _mount_args(paths: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """docker run の -v 引数と、サンドボックス内で見えるパスへの対応を返す。

    paths は {役割: 本体側のパス}。共有ボリューム方式ではボリューム全体を読み取り専用で
    1 回だけマウントし、各パスをボリューム内の相対パスで読み替える。
    bind mount 方式（ホスト実行）では従来どおり各パスを個別に /sandbox/<役割> へ渡す。
    """
    config = _share_config()
    if config is None:
        args: list[str] = []
        mapped: dict[str, str] = {}
        for role, local in paths.items():
            target = f"/sandbox/{role}"
            args += ["-v", f"{local}:{target}:ro"]
            mapped[role] = target
        return args, mapped

    share_dir, volume = config
    mapped = {}
    for role, local in paths.items():
        rel = os.path.relpath(local, share_dir)
        if rel.startswith(".."):
            raise RuntimeError(f"サンドボックス入力が共有ディレクトリの外にあります: {local}")
        mapped[role] = posixpath.join(_SANDBOX_SHARE_MOUNT, *rel.split(os.sep))
    return ["-v", f"{volume}:{_SANDBOX_SHARE_MOUNT}:ro"], mapped


def _nan_if_none(value: Optional[float]) -> float:
    return float("nan") if value is None else float(value)


def _returns_from_payload(payload: Optional[dict]) -> Optional[pd.Series]:
    """サンドボックスが返したポートフォリオ日次リターンを Series に戻す（無ければ None）。"""
    if not payload:
        return None
    return pd.Series(
        [float(v) for v in payload["values"]],
        index=pd.to_datetime(payload["dates"]),
        dtype=float,
    )


def prepare_sandbox_data(
    data_by_symbol: dict[str, pd.DataFrame],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[str, str]:
    """バッチ全体で使い回す共有データディレクトリ/windowsファイルを1回だけ作る。

    呼び出し元が一晩のバッチにつき1回だけ呼び、返り値のパスを
    run_sandboxed_evaluation に渡す。呼び出し元は使用後にディレクトリを
    削除する責務を持つ（tempfile.TemporaryDirectory 等で管理）。
    """
    workspace = _workspace_dir()
    data_dir = tempfile.mkdtemp(prefix="factory_sandbox_data_", dir=workspace)
    for symbol, df in data_by_symbol.items():
        df.to_parquet(os.path.join(data_dir, f"{symbol}.parquet"))

    windows_fd, windows_path = tempfile.mkstemp(
        prefix="factory_sandbox_windows_", suffix=".json", dir=workspace
    )
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

    with tempfile.TemporaryDirectory(
        prefix="factory_sandbox_src_", dir=_workspace_dir()
    ) as src_dir:
        source_path = os.path.join(src_dir, "candidate.py")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source_code)
        mount_args, mapped = _mount_args(
            {"src": src_dir, "data": shared_data_dir, "windows.json": windows_file}
        )

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
            # .env はサンドボックスコンテナにマウントされないため、ホスト側の実効値を
            # 明示的に渡さないとコンテナ内は config/settings.py のデフォルト（3）に
            # 固定されてしまう。ホスト側で FACTORY_GATE_MIN_TRADES_PER_SYMBOL を
            # 変更しても Claude 生成候補だけ旧閾値で集計される不整合を防ぐ（#625）。
            "-e",
            f"FACTORY_GATE_MIN_TRADES_PER_SYMBOL={FACTORY_GATE_MIN_TRADES_PER_SYMBOL}",
            *mount_args,
            image,
            "python",
            "scripts/sandbox_evaluate_rule.py",
            "--source-file",
            posixpath.join(mapped["src"], "candidate.py"),
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
            mapped["data"],
            "--windows-file",
            mapped["windows.json"],
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

        if payload.get("status") == "input_missing":
            # 候補のコードではなく受け渡しの問題。修復しても直らない（#757）
            return SandboxRunResult(
                kind="infra_error",
                infra_detail=f"サンドボックスが入力を読めません: {payload.get('detail', '')}",
                environment_broken=True,
            )

        if payload.get("status") == "error":
            return SandboxRunResult(
                kind="repairable",
                repair_detail=f"サンドボックス実行時エラー: {payload.get('traceback', '')[-2000:]}",
            )

        ev = payload["evaluation"]
        # n_symbols_with_signal / n_effective_symbols / avg_trades_per_symbol は
        # サンドボックスコンテナ（固定イメージ FACTORY_SANDBOX_IMAGE）がホストより
        # 古いバージョンだと出力しない可能性がある。旧イメージ由来のペイロードで
        # KeyError にしてバッチ全体を落とさないよう、dataclass 既定値と同じ値で
        # .get() フォールバックする（#625）。ただしこのフォールバックは
        # 「n_effective_symbols=0 でゲート不合格」という、ルール自体が悪いのか
        # イメージが古いのか見分けがつかない事故を静かに起こしうるため、
        # フォールバックが発動した事実を必ずログに残す。
        missing_keys = [
            key
            for key in ("n_symbols_with_signal", "n_effective_symbols", "avg_trades_per_symbol")
            if key not in ev
        ]
        if missing_keys:
            logger.warning(
                "[factory] サンドボックス出力に鍵が欠落しています（サンドボックスイメージが"
                "ホストより古い可能性が高い）: missing_keys=%s",
                missing_keys,
            )
        portfolio_keys = ("portfolio_sharpe_ratio", "portfolio_max_drawdown", "portfolio_returns")
        if any(key not in ev for key in portfolio_keys):
            # ゲートはポートフォリオ日次リターンで判定する。欠けていれば検定できず
            # 不合格に倒れるため、イメージの版ずれを疑えるようログに残す
            logger.warning(
                "[factory] サンドボックス出力にポートフォリオ指標が欠落しています"
                "（サンドボックスイメージがホストより古い可能性が高い）。"
                "この候補はゲートで判定不能として不合格になります: missing_keys=%s",
                [key for key in portfolio_keys if key not in ev],
            )
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
            n_symbols_with_signal=ev.get("n_symbols_with_signal", 0),
            n_effective_symbols=ev.get("n_effective_symbols", 0),
            avg_trades_per_symbol=ev.get("avg_trades_per_symbol", 0.0),
            portfolio_sharpe_ratio=_nan_if_none(ev.get("portfolio_sharpe_ratio")),
            portfolio_max_drawdown=_nan_if_none(ev.get("portfolio_max_drawdown")),
            portfolio_returns=_returns_from_payload(ev.get("portfolio_returns")),
        )
        return SandboxRunResult(kind="gate_evaluated", evaluation=evaluation)
