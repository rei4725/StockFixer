"""
correlation_risk.py — 相関ベースのポートフォリオリスク管理

保有銘柄間の動的相関行列（ローリングウィンドウ）を計算し、
実効分散度（ENC）が閾値を下回った場合に新規エントリーをブロックする。

ENC 式: ENC = N / (1 + avg_corr * (N-1))
  - N: 保有銘柄数
  - avg_corr: 銘柄間の平均絶対相関係数
  - ENC=N のとき完全分散、ENC=1 のとき完全相関
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.trading.types import CorrelationGateResult
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CorrelationDataUnavailable(Exception):
    """相関の計算に使う終値を DB から読めなかった（SQL の誤り・DB の不調など）。

    データが足りないだけ（新しい銘柄・行数不足）は空の表で返し、この例外にはしない。
    """


def _load_recent_returns(symbols: list[str], market: str, window: int) -> pd.DataFrame:
    """market_data_raw から直近 window 日分の日足終値リターンを取得する。

    stock_features はラグ特徴量（Close_lag1 等）しか保存しないため生の終値を持たない。
    終値は生 OHLCV の market_data_raw.close から読み、日付（ts）で銘柄を揃える。
    """
    if not symbols:
        return pd.DataFrame()

    placeholders = ", ".join(["%s"] * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT symbol, ts, close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
            FROM market_data_raw
            WHERE market = %s
              AND timeframe = %s
              AND symbol IN ({placeholders})
              AND close IS NOT NULL
        )
        SELECT symbol, ts, close
        FROM ranked
        WHERE rn <= %s
        ORDER BY symbol, ts
    """

    try:
        with _db_connection() as con:
            df = pd.read_sql(query, con, params=[market, "1d"] + symbols + [window + 1])
    except Exception as e:
        # 空の表で返すとデータ不足と区別がつかず、ゲートが黙って素通しになる（#809）
        logger.error("相関計算用データ取得失敗: %s", e, exc_info=True)
        raise CorrelationDataUnavailable(repr(e)[:300]) from e

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index="ts", columns="symbol", values="close")
    # 日足の短い銘柄を残すと dropna() が全銘柄の行を落とし、ゲートが素通しになる（#1001）
    min_returns = max(2, window / 2)
    return_counts = pivot.count() - 1
    short_symbols = list(return_counts[return_counts < min_returns].index)
    if short_symbols:
        logger.warning(
            "[corr] 日足の本数が足りない銘柄を相関の計算から外す: %s (必要なリターン数>=%s)",
            short_symbols,
            min_returns,
        )
        pivot = pivot.drop(columns=short_symbols)
    if pivot.empty:
        return pd.DataFrame()
    returns = pivot.pct_change().dropna()
    # 日足の止まった銘柄は古い日付で本数がそろい、dropna() が共通の行を減らす（#1141）
    while len(returns) < min_returns and len(pivot.columns) > 1:
        last_dates = pd.Series({sym: pivot[sym].last_valid_index() for sym in pivot.columns})
        stale_symbols = list(last_dates[last_dates < last_dates.max()].index)
        if not stale_symbols:
            break
        logger.warning(
            "[corr] 日足の最後の日が古い銘柄を相関の計算から外す: %s (共通のリターン数=%d < %s)",
            stale_symbols,
            len(returns),
            min_returns,
        )
        pivot = pivot.drop(columns=stale_symbols).dropna(how="all")
        returns = pivot.pct_change().dropna()
    # 窓の途中で日足の欠けた銘柄は本数も最後の日もそろい、上の2つで外れない（#1254）
    while len(returns) < min_returns and len(pivot.columns) > 1:
        missing = pivot.iloc[-(window + 1) :].isna().sum()
        gap_symbols = list(missing[missing == missing.max()].index)
        if missing.max() == 0 or len(gap_symbols) == len(pivot.columns):
            break
        logger.warning(
            "[corr] 窓の中で日足の欠けが多い銘柄を相関の計算から外す: %s (欠けた日数=%d, 共通のリターン数=%d < %s)",
            gap_symbols,
            int(missing.max()),
            len(returns),
            min_returns,
        )
        pivot = pivot.drop(columns=gap_symbols).dropna(how="all")
        returns = pivot.pct_change().dropna()
    if len(returns) < min_returns and len(pivot.columns) > 1:
        # 数本の相関は大きく振れ、1本なら NaN になる。判定に使わずデータ不足として返す
        logger.warning(
            "[corr] 共通のリターンが足りないため相関の表を返さない: %s (共通のリターン数=%d < %s)",
            list(pivot.columns),
            len(returns),
            min_returns,
        )
        return pd.DataFrame()
    return returns


def compute_enc(corr_matrix: pd.DataFrame) -> tuple[float, float]:
    """相関行列から ENC と平均絶対相関係数を計算する。

    Returns:
        (enc, avg_correlation) のタプル
    """
    n = len(corr_matrix)
    if n <= 1:
        return float(n), 0.0

    mask = ~np.eye(n, dtype=bool)
    off_diag = corr_matrix.values[mask]
    avg_corr = float(np.mean(np.abs(off_diag)))

    enc = n / (1.0 + avg_corr * (n - 1))
    return enc, avg_corr


def filter_correlated_candidates(
    candidate_symbols: list[str],
    existing_symbols: list[str],
    market: str,
    window: int = 60,
    threshold: float = 0.7,
) -> tuple[list[str], list[str]]:
    """候補銘柄から既存ポジションと高相関な銘柄を除外する。

    直近 window 日の日次リターンから各候補と既存ポジションとのペアワイズ絶対相関を計算し、
    threshold を超える候補をブロックする。

    Args:
        candidate_symbols: 新規エントリー候補の銘柄コードリスト
        existing_symbols: 既存ポジションの銘柄コードリスト
        market: マーケット識別子
        window: 相関計算に使うローリングウィンドウ日数
        threshold: ブロック閾値（この値を超えると新規エントリーをブロック）

    Returns:
        (allowed_symbols, blocked_symbols) のタプル
    """
    if not candidate_symbols or not existing_symbols:
        return list(candidate_symbols), []

    all_symbols = list(set(candidate_symbols) | set(existing_symbols))
    try:
        returns = _load_recent_returns(all_symbols, market, window)
    except CorrelationDataUnavailable as e:
        logger.error(
            "[corr] 相関のデータを読めないため候補を全部止める (candidates=%s): %s",
            candidate_symbols,
            e,
        )
        return [], list(candidate_symbols)

    if returns.empty:
        logger.warning(
            "[corr] ペアワイズ相関計算用データ不足のためスキップ (candidates=%s)", candidate_symbols
        )
        return list(candidate_symbols), []

    existing_in_data = [s for s in existing_symbols if s in returns.columns]
    if not existing_in_data:
        return list(candidate_symbols), []

    allowed: list[str] = []
    blocked: list[str] = []

    for sym in candidate_symbols:
        if sym not in returns.columns:
            allowed.append(sym)
            continue

        max_corr = 0.0
        for existing in existing_in_data:
            if existing == sym:
                continue
            corr_val = returns[sym].corr(returns[existing])
            if not np.isnan(corr_val):
                max_corr = max(max_corr, abs(corr_val))

        if max_corr > threshold:
            logger.info(
                "[corr] ペアワイズ相関でブロック: %s (max_corr=%.2f > threshold=%.2f)",
                sym,
                max_corr,
                threshold,
            )
            blocked.append(sym)
        else:
            allowed.append(sym)

    if blocked:
        logger.warning(
            "[corr] ペアワイズ相関フィルタ: %d 銘柄ブロック, %d 銘柄通過 (threshold=%.2f)",
            len(blocked),
            len(allowed),
            threshold,
        )

    return allowed, blocked


def evaluate_correlation_gate(
    symbols: list[str],
    market: str,
    window: int = 20,
    enc_threshold: float = 2.0,
) -> CorrelationGateResult:
    """保有銘柄の相関ゲートを評価する。

    Args:
        symbols: 保有銘柄コードのリスト
        market: マーケット識別子
        window: ローリングウィンドウ日数
        enc_threshold: ENC の最低許容値（この値を下回ると新規エントリーをブロック）

    Returns:
        CorrelationGateResult — is_allowed=False の場合、新規買いをブロックすべき
    """
    n = len(symbols)

    if n < 2:
        return CorrelationGateResult(
            is_allowed=True,
            enc=float(n),
            enc_threshold=enc_threshold,
            avg_correlation=0.0,
            n_symbols=n,
            symbols=list(symbols),
        )

    try:
        returns = _load_recent_returns(symbols, market, window)
    except CorrelationDataUnavailable as e:
        failure = f"相関のデータを読めないため新規エントリーを止める: {e}"
        logger.error("[corr] %s", failure)
        return CorrelationGateResult(
            is_allowed=False,
            enc=float(n),
            enc_threshold=enc_threshold,
            avg_correlation=0.0,
            n_symbols=n,
            symbols=list(symbols),
            reason=failure,
        )

    if returns.empty or len(returns.columns) < 2:
        logger.warning("[corr] リターンデータ不足のため相関ゲートをスキップ (symbols=%s)", symbols)
        return CorrelationGateResult(
            is_allowed=True,
            enc=float(n),
            enc_threshold=enc_threshold,
            avg_correlation=0.0,
            n_symbols=n,
            symbols=list(symbols),
            reason="リターンデータ不足のためスキップ",
        )

    corr_matrix = returns.corr()
    enc, avg_corr = compute_enc(corr_matrix)

    is_allowed = enc >= enc_threshold
    reason: Optional[str] = None
    if not is_allowed:
        reason = (
            f"ENC={enc:.2f} < 閾値={enc_threshold:.2f} " f"(銘柄数={n}, 平均相関={avg_corr:.2f})"
        )
        logger.warning("[corr] 相関リスク上昇検知: %s → 新規エントリーをブロック", reason)
    else:
        logger.info(
            "[corr] 相関ゲート通過: ENC=%.2f >= 閾値=%.2f (avg_corr=%.2f, N=%d)",
            enc,
            enc_threshold,
            avg_corr,
            n,
        )

    return CorrelationGateResult(
        is_allowed=is_allowed,
        enc=enc,
        enc_threshold=enc_threshold,
        avg_correlation=avg_corr,
        n_symbols=n,
        symbols=list(symbols),
        reason=reason,
    )
