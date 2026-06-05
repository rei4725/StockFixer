"""長期トレンド・スクリーナー本体

米国株の生 OHLCV（DuckDB `market_data_raw`）を使い、長期上昇トレンドにある
multibagger 候補を抽出・ランキングする。

注意: 当日終値は `market_data_raw` から取得する。`stock_features` は ML 用の
ラグ・派生特徴量テーブルで生の Close 列を持たないため使わない。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.screening.types import TrendCandidate
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.db.market_data import load_all_raw_ohlcv_symbols, load_raw_ohlcv
from src.utils.logger import get_logger

logger = get_logger(__name__)

# スコア合成の重み（合計1.0）。相対強度を最重視し、トレンドの質を補助的に評価する。
# - 相対強度: multibagger は強いモメンタムが起点になることが多いため最重視。
# - 高値接近度: 52週高値圏は需給が良好で上値抵抗が少ない。
# - トレンド強度: 200日線からの乖離が大きいほど長期上昇の勢いが強い。
# 後で最適化できるよう定数として切り出す。
_W_REL_STRENGTH = 0.50
_W_PROXIMITY = 0.25
_W_TREND = 0.25


def _compute_metrics(df: pd.DataFrame) -> Optional[dict]:
    """1銘柄の長期トレンド指標を Close 列から計算する。

    計算に必要なデータが不足している場合は None を返す。
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].astype(float).reset_index(drop=True)
    if len(close) < 252:
        return None

    sma200 = close.rolling(200).mean()
    high52 = close.rolling(252).max()

    last_close = float(close.iloc[-1])
    sma200_last = sma200.iloc[-1]
    high52_last = high52.iloc[-1]
    if pd.isna(sma200_last) or pd.isna(high52_last) or high52_last <= 0:
        return None

    # 200日SMAが直近20日で上向きか
    sma200_prev = sma200.iloc[-21]
    sma200_rising = bool(not pd.isna(sma200_prev) and sma200_last > sma200_prev)

    above_200dma = bool(last_close > sma200_last)
    dist_from_52w_high = last_close / float(high52_last) - 1.0
    trend_strength = last_close / float(sma200_last) - 1.0

    return_6m = last_close / float(close.iloc[-126]) - 1.0
    return_12m = last_close / float(close.iloc[-252]) - 1.0

    avg_volume = float(df["Volume"].mean()) if "Volume" in df.columns else 0.0

    return {
        "close": last_close,
        "dist_from_52w_high": dist_from_52w_high,
        "above_200dma": above_200dma,
        "sma200_rising": sma200_rising,
        "return_6m": return_6m,
        "return_12m": return_12m,
        "avg_volume": avg_volume,
        "trend_strength": trend_strength,
    }


def _truncate_as_of(df: pd.DataFrame, as_of: Optional[str]) -> pd.DataFrame:
    """評価基準日 as_of 以前の行のみに切り詰める（ルックアヘッド防止）。

    load_raw_ohlcv は日付を index（"Date"）に持つため index で比較する。
    後方互換のため "date" 列がある場合はそちらも使える。
    as_of が None、または df が空の場合はそのまま返す。
    """
    if as_of is None or df is None or df.empty:
        return df
    if "date" in df.columns:
        mask = pd.to_datetime(df["date"]) <= pd.Timestamp(as_of)
        return df[mask]
    idx = pd.to_datetime(df.index)
    return df[idx <= pd.Timestamp(as_of)]


def screen_trend_candidates(
    market: str = "us",
    top_n: int = 30,
    max_dist_from_high: float = 0.25,  # 52週高値から -25% 以内
    rel_strength_pct: float = 0.80,  # 相対強度 上位20%（=80パーセンタイル以上）
    min_avg_volume: float = 100_000,  # 最低平均出来高
    min_price: float = 5.0,  # 最低株価
    min_data_days: int = 252,  # 最低データ日数
    as_of: Optional[str] = None,  # 評価基準日（YYYY-MM-DD）。指定時はこの日以前のみ使用
) -> list[TrendCandidate]:
    """長期上昇トレンドにある multibagger 候補を抽出・ランキングする。

    Args:
        as_of: 評価基準日。指定するとこの日（含む）までの終値のみで判定する。
            長期バックテスト(#431)が過去時点のスクリーンを再現するために使う。

    Returns:
        スコア降順の TrendCandidate リスト（最大 top_n 件）。該当なしなら空リスト。
    """
    symbols = [(m, s) for (m, s) in load_all_raw_ohlcv_symbols() if m == market]
    logger.info(f"スクリーニング対象: {len(symbols)} 銘柄 ({market})")

    rows: list[dict] = []
    for m, symbol in symbols:
        df = load_raw_ohlcv(m, symbol)
        df = _truncate_as_of(df, as_of)
        if df is None or len(df) < min_data_days:
            continue

        metrics = _compute_metrics(df)
        if metrics is None:
            continue

        # 流動性・株価フィルタ
        if metrics["close"] < min_price or metrics["avg_volume"] < min_avg_volume:
            continue

        # ハードゲート: 200日線上 ＋ 上向き ＋ 52週高値圏
        if not (metrics["above_200dma"] and metrics["sma200_rising"]):
            continue
        if metrics["dist_from_52w_high"] < -max_dist_from_high:
            continue

        # 相対強度の指標は 6m と 12m の平均リターン
        metrics["rs_metric"] = (metrics["return_6m"] + metrics["return_12m"]) / 2.0
        metrics["market"] = m
        metrics["symbol"] = symbol
        rows.append(metrics)

    if not rows:
        logger.warning("スクリーニング結果が空です（ゲート通過銘柄なし）")
        return []

    universe = pd.DataFrame(rows)

    # 相対強度のユニバース内パーセンタイル順位で上位のみ残す
    universe["rs_rank"] = universe["rs_metric"].rank(pct=True)
    universe = universe[universe["rs_rank"] >= rel_strength_pct]
    if universe.empty:
        logger.warning("相対強度フィルタ通過銘柄なし")
        return []

    # 合成スコア: 各指標のユニバース内順位（0〜1正規化）を加重平均
    rel_rank = universe["rs_metric"].rank(pct=True)
    prox_rank = universe["dist_from_52w_high"].rank(pct=True)  # 0に近いほど高評価
    trend_rank = universe["trend_strength"].rank(pct=True)
    universe = universe.assign(
        score=(rel_rank * _W_REL_STRENGTH + prox_rank * _W_PROXIMITY + trend_rank * _W_TREND)
    )

    universe = universe.sort_values("score", ascending=False).head(top_n)

    candidates = [
        TrendCandidate(
            market=row["market"],
            symbol=row["symbol"],
            score=float(row["score"]),
            close=float(row["close"]),
            dist_from_52w_high=float(row["dist_from_52w_high"]),
            above_200dma=bool(row["above_200dma"]),
            sma200_rising=bool(row["sma200_rising"]),
            return_6m=float(row["return_6m"]),
            return_12m=float(row["return_12m"]),
            avg_volume=float(row["avg_volume"]),
        )
        for _, row in universe.iterrows()
    ]

    logger.info(f"スクリーニング完了: {len(candidates)} 銘柄を選定 ({market})")
    return candidates


def save_candidates(candidates: list[TrendCandidate], market: str) -> str:
    """候補リストを CSV に保存しパスを返す。"""
    out_dir = ensure_dir(f"{get_results_dir()}/screening")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/trend_candidates_{market}_{timestamp}.csv"

    df = pd.DataFrame([c.__dict__ for c in candidates])
    df.to_csv(path, index=False)
    logger.info(f"候補を保存: {path}")
    return path
