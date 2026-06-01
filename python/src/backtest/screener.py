"""
高ボラティリティ銘柄スクリーナー

ウォッチリストからATR（ボラティリティ）と出来高が高い銘柄を自動選定する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.data_port import get_backtest_data_port
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WATCHLIST_PATH = Path(__file__).parent.parent.parent.parent / "config" / "watchlist.json"


def _load_watchlist_symbols(market: str) -> list[str]:
    with open(_WATCHLIST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get(market, []))


def _calc_volatility_score(df: pd.DataFrame) -> dict[str, Any]:
    """ATR% と平均出来高からスコアを算出する。"""
    if df.empty or len(df) < 20:
        return {"avg_volume": 0.0, "avg_atr_pct": 0.0, "score": 0.0}

    df = get_backtest_data_port().add_technical_indicators(df)

    avg_volume = float(df["Volume"].mean())
    avg_atr_pct = float((df["atr"] / df["Close"]).mean()) if "atr" in df.columns else 0.0

    return {
        "avg_volume": avg_volume,
        "avg_atr_pct": avg_atr_pct,
        "score": 0.0,  # 正規化後に設定
    }


def screen_volatile_symbols(
    market: str,
    top_n: int = 10,
    screen_start: str = "2024-01-01",
    screen_end: str = "2024-12-31",
    min_data_days: int = 100,
) -> list[str]:
    """
    ウォッチリストから高ボラティリティ・高出来高の銘柄を選定する。

    Args:
        market: マーケット識別子（"jp" or "us"）
        top_n: 返す銘柄数
        screen_start: スクリーニング期間の開始日
        screen_end: スクリーニング期間の終了日
        min_data_days: 最低必要データ日数

    Returns:
        銘柄コードのリスト（スコア降順）
    """
    symbols = _load_watchlist_symbols(market)
    logger.info(f"スクリーニング対象: {len(symbols)} 銘柄 ({market})")

    rows = []
    for symbol in symbols:
        ticker = get_ticker(market, symbol)
        try:
            df = get_backtest_data_port().download(ticker, start=screen_start, end=screen_end)
            if df is None or len(df) < min_data_days:
                continue
            stats = _calc_volatility_score(df)
            stats["symbol"] = symbol
            rows.append(stats)
        except Exception as exc:
            logger.debug(f"スキップ [{symbol}]: {exc}")

    if not rows:
        logger.warning("スクリーニング結果が空です")
        return []

    result = pd.DataFrame(rows)

    # 正規化スコア = 出来高rank * 0.4 + ATR%rank * 0.6（ボラ重視）
    result["vol_rank"] = result["avg_volume"].rank(pct=True)
    result["atr_rank"] = result["avg_atr_pct"].rank(pct=True)
    result["score"] = result["vol_rank"] * 0.4 + result["atr_rank"] * 0.6

    result = result.sort_values("score", ascending=False).head(top_n)

    logger.info(f"スクリーニング完了: Top {top_n} 銘柄を選定")
    for _, row in result.iterrows():
        logger.info(
            f"  {row['symbol']:>6}  出来高={row['avg_volume']:>12,.0f}"
            f"  ATR%={row['avg_atr_pct']:.3%}  score={row['score']:.3f}"
        )

    return list(result["symbol"].tolist())
