"""
見張りの5日に、旧 production の写しで同じ行を予想させ、新しいモデルと同じ窓の物差しを出す（#1729）。

戻すかは、入れ替え後の5日の新しいモデルの実測と、同じ (market, symbol, 基準の日) の行で
旧 production が出した予想とを比べて決める。取っておいた20日の旧 production の数字とは比べない
（窓が違うと、相場の向きで上下が決まってしまうため）。

予想は取っておいた期間の比べと同じ芯（ensemble_diff_ratio）と同じ行の作り方（feature_row）で出す。
重みは本番の名前で引く。基準の日は prediction_accuracy の突き合わせと同じく _base_bar_date で決め、
その日以前の最後の特徴量の行を使う。
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd

from src.prediction.backtest_promotion.holdout import _close_by_day
from src.prediction.backtest_promotion.judge import MIN_COVERAGE
from src.prediction.backtest_promotion.state import SWAP_PRODUCTION_NAMES, _production_for_backup
from src.prediction.baseline_metrics import baseline_metrics
from src.prediction.db import load_model_weights
from src.prediction.predict_unified import ensemble_diff_ratio, feature_row
from src.prediction.prediction_pipeline import _base_bar_date
from src.prediction.unified_model_pipeline import load_unified_model
from src.utils.db import load_raw_ohlcv, load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

_HORIZON = 1
NO_BACKUPS = "旧 production の写しが無い"
UNREADABLE_BACKUPS = "旧 production の写しが読めない"
MISSING_BACKUPS = "旧 production の写しがそろわない"
NO_PREDICTIONS = "同じ行の旧 production の予想が出ない"
SHORT_PREDICTIONS = "同じ行の旧 production の予想が足りない"


def load_backup_models(backups: list[str]) -> list[tuple[str, Any]]:
    """写しから旧 production のモデルを読む。（重みを引く本番の名前, モデル）の並び。

    Raises:
        ValueError: 本番の写しでないパスがあるとき
        FileNotFoundError: 写しが無いとき
    """
    models = []
    for path in backups:
        production = _production_for_backup(path)
        name = os.path.splitext(os.path.basename(path))[0]
        models.append((production, load_unified_model(name, model_dir=os.path.dirname(path))))
    return models


def _old_ratio(
    latest_X: pd.DataFrame, price: float, market: str, symbol: str, models: list
) -> Optional[float]:
    """旧 production の予想の変化率。全モデルが例外か、使えるモデルが無ければ None。"""
    try:
        outcome = ensemble_diff_ratio(
            latest_X, price, market, symbol, _HORIZON, models, weights_fn=load_model_weights
        )
    except RuntimeError:
        logger.error(
            "旧 production の予想が全モデルで失敗したため行を飛ばす: market=%s symbol=%s",
            market,
            symbol,
            exc_info=True,
        )
        return None
    return None if outcome is None else outcome.diff_ratio


def _predict_symbol(
    market: str, symbol: str, group: pd.DataFrame, models: list, skipped: dict[str, int]
) -> list[dict]:
    """1銘柄の行について、旧 production の予想を足した行を返す。"""
    df = load_stock_features(market, symbol, raise_on_error=True)
    if df is None or df.empty or "date" not in df.columns:
        skipped["no_features"] += len(group)
        return []
    row_days = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    bases = [pd.Timestamp(_base_bar_date(str(p), market)) for p in group["predicted_at"]]
    closes = _close_by_day(load_raw_ohlcv(market, symbol, start_date=min(bases)))
    out = []
    for (_, row), base in zip(group.iterrows(), bases):
        candidates = [i for i, d in enumerate(row_days) if pd.notna(d) and d <= base]
        if not candidates:
            skipped["no_features"] += 1
            continue
        pos = candidates[-1]
        price = closes.get(pd.Timestamp(row_days.iloc[pos]))
        if price is None:
            skipped["no_price"] += 1
            continue
        old = _old_ratio(feature_row(df, pos, market), price, market, symbol, models)
        if old is None:
            skipped["no_prediction"] += 1
            continue
        out.append(
            {
                "actual_ratio": float(row["actual_ratio"]),
                "new_ratio": float(row["predicted_ratio"]),
                "old_ratio": old,
            }
        )
    return out


def measure_old_production(backups: list[str], rows: pd.DataFrame) -> dict:
    """見張りの5日の行について、旧 production と新しいモデルの同じ行の物差しを出す。

    Args:
        backups: 旧 production の写しのパス（見張りの backups）
        rows: 入れ替え後の最初の5日の、採点済みの production の行
            （market, symbol, predicted_at, predicted_ratio, actual_ratio）

    Returns:
        dict。old・new は同じ行の baseline_metrics()、n_compared は比べた行の数。
        測れない回は old・new が None で、reason に理由（その回は戻さない）。

    Raises:
        Exception: 特徴量・値段の読み込みの失敗（DB の例外。見張りは残して次の夜にやり直す）
    """
    result: dict = {
        "n_watch_rows": int(len(rows)),
        "n_compared": 0,
        "need_rows": len(rows) * MIN_COVERAGE,
        "old": None,
        "new": None,
        "reason": None,
        "skipped": {"no_features": 0, "no_price": 0, "no_prediction": 0},
    }
    if not backups:
        result["reason"] = NO_BACKUPS
        return result
    try:
        models = load_backup_models(backups)
    except Exception as e:
        logger.error("旧 production の写しを読めない: %s", backups, exc_info=True)
        result["reason"] = f"{UNREADABLE_BACKUPS}（{type(e).__name__}: {e}）"
        return result
    missing = sorted(set(SWAP_PRODUCTION_NAMES) - {name for name, _ in models})
    if missing:
        result["reason"] = f"{MISSING_BACKUPS}（無い: {missing}）"
        return result

    compared: list[dict] = []
    for (market, symbol), group in rows.groupby(["market", "symbol"], sort=True):
        compared += _predict_symbol(str(market), str(symbol), group, models, result["skipped"])
    result["n_compared"] = len(compared)
    if not compared:
        result["reason"] = NO_PREDICTIONS
        return result
    if len(compared) < result["need_rows"]:
        result["reason"] = (
            f"{SHORT_PREDICTIONS}（{len(compared)} 行、必要 {result['need_rows']:g} 行以上）"
        )
        return result
    table = pd.DataFrame(compared)
    result["old"] = baseline_metrics(table.rename(columns={"old_ratio": "predicted_ratio"}))
    result["new"] = baseline_metrics(table.rename(columns={"new_ratio": "predicted_ratio"}))
    return result
