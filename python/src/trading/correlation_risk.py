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


def _load_recent_returns(symbols: list[str], market: str, window: int) -> pd.DataFrame:
    """stock_features から直近 window 日分の終値リターンを取得する。"""
    if not symbols:
        return pd.DataFrame()

    placeholders = ", ".join(["%s"] * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT symbol, row_num, close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY row_num DESC) AS rn
            FROM stock_features
            WHERE market = %s
              AND symbol IN ({placeholders})
              AND close IS NOT NULL
        )
        SELECT symbol, row_num, close
        FROM ranked
        WHERE rn <= %s
        ORDER BY symbol, row_num
    """

    try:
        with _db_connection() as con:
            df = pd.read_sql(query, con, params=[market] + symbols + [window + 1])
    except Exception as e:
        logger.error("相関計算用データ取得失敗: %s", e, exc_info=True)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index="row_num", columns="symbol", values="close")
    returns = pivot.pct_change().dropna()
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
    returns = _load_recent_returns(all_symbols, market, window)

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

    returns = _load_recent_returns(symbols, market, window)

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
