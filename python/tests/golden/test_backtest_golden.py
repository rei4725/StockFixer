"""Golden test: バックテスト結果のリグレッション防止

固定 seed・固定入力で simulate_trading を実行し、
結果（取引ログ・指標）のハッシュが変化しないことを保証する。

golden ファイルの更新方法:
    UPDATE_GOLDEN=1 python -m pytest tests/golden/ -v
"""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import Backtester

GOLDEN_DIR = Path(__file__).parent / "data"
GOLDEN_FILE = GOLDEN_DIR / "backtest_golden.json"

_SEED = 42
_N_DAYS = 30
_INITIAL_CASH = 1_000_000
_FEE_RATE = 0.001
_SLIPPAGE = 0.0005


def _make_price_df() -> pd.DataFrame:
    rng = np.random.default_rng(seed=_SEED)
    dates = pd.date_range("2024-01-01", periods=_N_DAYS, freq="B")
    close = 100.0 + np.cumsum(rng.uniform(-2.0, 2.0, _N_DAYS))
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.02,
            "Low": close * 0.97,
            "Close": close,
            "Volume": rng.integers(1000, 5000, _N_DAYS).astype(float),
        },
        index=dates,
    )


def _make_signal(index: pd.DatetimeIndex) -> pd.Series:
    pattern = [1, 0, 0, -1, 0, 1, 0, -1, 0, 0]
    signals = [pattern[i % len(pattern)] for i in range(len(index))]
    return pd.Series(signals, index=index)


def _make_backtester() -> Backtester:
    return Backtester(
        model_manager=MagicMock(),
        signal_generator=MagicMock(),
        data_loader=MagicMock(),
        start_date=None,
        end_date=None,
        market="jp",
        symbol="GOLDEN",
        initial_cash=_INITIAL_CASH,
        fee_rate=_FEE_RATE,
        slippage=_SLIPPAGE,
    )


def _serialize_results(result_df: pd.DataFrame, metrics: dict) -> dict:
    """結果を JSON シリアライズ可能な dict に変換する。"""
    trade_rows = []
    for _, row in result_df.iterrows():
        entry: dict = {}
        for col in ["date", "action", "price", "qty", "cash"]:
            if col not in row.index:
                continue
            val = row[col]
            if isinstance(val, pd.Timestamp):
                val = str(val.date())
            elif isinstance(val, float):
                val = round(float(val), 6)
            elif hasattr(val, "item"):
                val = val.item()
            entry[col] = val
        trade_rows.append(entry)

    metrics_out: dict = {}
    for k, v in metrics.items():
        if v is None:
            metrics_out[k] = None
        elif isinstance(v, float):
            metrics_out[k] = round(float(v), 8)
        elif isinstance(v, int):
            metrics_out[k] = int(v)
        else:
            metrics_out[k] = v

    return {"trade_log": trade_rows, "metrics": metrics_out}


def _compute_hash(data: dict) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class TestBacktestGolden:
    def test_simulate_trading_golden(self):
        """固定入力でのバックテスト結果ハッシュが変化しないことを検証する。

        バックテストロジックを修正した際に意図しない数値変化を検知するための
        リグレッションテスト。golden ファイルは UPDATE_GOLDEN=1 で更新する。
        """
        price_df = _make_price_df()
        signal = _make_signal(price_df.index)
        bt = _make_backtester()

        result_df, metrics = bt.simulate_trading(price_df, signal)

        data = _serialize_results(result_df, metrics)
        actual_hash = _compute_hash(data)

        if os.environ.get("UPDATE_GOLDEN", "0") == "1":
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            with open(GOLDEN_FILE, "w", encoding="utf-8") as f:
                json.dump({"hash": actual_hash, "data": data}, f, indent=2, ensure_ascii=False)
            pytest.skip(f"Golden file updated: {GOLDEN_FILE}")
            return

        assert GOLDEN_FILE.exists(), (
            f"Golden file not found: {GOLDEN_FILE}\n"
            "Run with UPDATE_GOLDEN=1 to create it:\n"
            "  UPDATE_GOLDEN=1 python -m pytest tests/golden/ -v"
        )
        with open(GOLDEN_FILE, encoding="utf-8") as f:
            golden = json.load(f)

        assert actual_hash == golden["hash"], (
            f"Backtest golden test FAILED: hash mismatch\n"
            f"expected : {golden['hash']}\n"
            f"actual   : {actual_hash}\n"
            "If this change is intentional, update the golden file:\n"
            "  UPDATE_GOLDEN=1 python -m pytest tests/golden/ -v"
        )
