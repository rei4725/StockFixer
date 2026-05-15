"""
資本配分エンジン（R-301）

複数候補銘柄の予測値・信頼度・相関・レジームを統合して
ポートフォリオ全体の保有比率を最適化する。

配分アルゴリズム:
    1. 基本配分: prediction の diff_ratio（シグナル強度）に比例
    2. 信頼度補正: prediction_accuracy の directional_accuracy で重み付け
    3. 相関ペナルティ: 相関係数が高い銘柄ペアは配分を抑制
    4. レジーム補正: bear レジームは配分を縮小（bull: 1.0, range: 0.8, bear: 0.5）
    5. 制約適用: MAX_POSITION_RATE / MAX_POSITIONS / MAX_SECTOR_POSITIONS

出力は `AllocationResult` dataclass のリストで、
portfolio_backtest.py と order_execution_pipeline.py から再利用できる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config.settings import MAX_POSITION_RATE, MAX_POSITIONS, MAX_SECTOR_POSITIONS
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger
from src.utils.sector_constraints import filter_by_sector_cap, get_symbol_sector

logger = get_logger(__name__)

# レジーム別のスケール係数
_REGIME_SCALE: dict[str, float] = {
    "bull": 1.0,
    "range": 0.8,
    "bear": 0.5,
    "unknown": 0.8,
}

# 相関ペナルティの感度: 相関係数がこの値を超えると配分を抑制し始める
_CORR_PENALTY_THRESHOLD = 0.7
_CORR_WINDOW_DAYS = 20


@dataclass
class AllocationResult:
    """単一銘柄の配分結果。"""

    market: str
    symbol: str
    weight: float  # 0.0〜1.0（ポートフォリオ比率、合計1.0以下）
    position_size_ratio: float  # MAX_POSITION_RATE に対する比率
    signal_strength: float  # diff_ratio（生のシグナル）
    confidence: float  # directional_accuracy（0〜1）
    regime: str  # "bull" / "bear" / "range" / "unknown"
    sector: Optional[str] = None
    allocation_reason: str = ""
    allocated_capital: float = field(default=0.0, repr=False)  # 配分金額（円）


def _load_confidence_map(market: str, symbols: list[str]) -> dict[str, float]:
    """prediction_accuracy から銘柄ごとの方向正解率を取得する。

    データが存在しない銘柄は 0.5（ランダム水準）を返す。
    """
    if not symbols:
        return {}

    placeholders = ", ".join("?" * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT symbol,
                   AVG(CAST(direction_match AS INTEGER)) AS dir_acc,
                   COUNT(*) AS n
            FROM prediction_accuracy
            WHERE market = ?
              AND symbol IN ({placeholders})
              AND horizon = 1
            GROUP BY symbol
        )
        SELECT symbol, dir_acc, n FROM ranked
    """
    try:
        with _db_connection() as con:
            rows = con.execute(query, [market] + symbols).fetchall()
    except Exception as e:
        logger.error("信頼度データ取得失敗: %s", e, exc_info=True)
        return {}

    result: dict[str, float] = {}
    for symbol, dir_acc, _n in rows:
        result[symbol] = float(dir_acc) if dir_acc is not None else 0.5
    return result


def _load_returns_for_corr(symbols: list[str], market: str) -> pd.DataFrame:
    """stock_features から直近20日分のリターンを取得し、相関行列計算に使う。"""
    if len(symbols) < 2:
        return pd.DataFrame()

    placeholders = ", ".join("?" * len(symbols))
    query = f"""
        WITH ranked AS (
            SELECT symbol, row_num, close,
                   ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY row_num DESC) AS rn
            FROM stock_features
            WHERE market = ?
              AND symbol IN ({placeholders})
              AND close IS NOT NULL
        )
        SELECT symbol, row_num, close
        FROM ranked
        WHERE rn <= ?
        ORDER BY symbol, row_num
    """
    try:
        with _db_connection() as con:
            df = con.execute(query, [market] + symbols + [_CORR_WINDOW_DAYS + 1]).df()
    except Exception as e:
        logger.error("相関計算用データ取得失敗: %s", e, exc_info=True)
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index="row_num", columns="symbol", values="close")
    returns = pivot.pct_change().dropna()
    return returns


def _detect_regime(market: str, regime_override: Optional[str]) -> str:
    """市場レジームを判定する。

    regime_override が指定された場合はその値を返す（テスト用）。
    指定がなければ stock_features から代表銘柄のデータで get_market_regime を呼ぶ。
    """
    if regime_override is not None:
        return regime_override

    # stock_features に市場インデックスを格納していないため、
    # 最も行数の多い銘柄のデータをプロキシとして使う。
    query = """
        SELECT symbol, COUNT(*) AS n
        FROM stock_features
        WHERE market = ?
          AND close IS NOT NULL
        GROUP BY symbol
        ORDER BY n DESC
        LIMIT 1
    """
    try:
        with _db_connection() as con:
            row = con.execute(query, [market]).fetchone()
    except Exception as e:
        logger.warning("レジーム判定用銘柄取得失敗: %s", e, exc_info=True)
        return "unknown"

    if row is None:
        return "unknown"

    proxy_symbol = row[0]
    ohlcv_query = """
        SELECT row_num AS "Date",
               open AS "Open", high AS "High", low AS "Low",
               close AS "Close", volume AS "Volume"
        FROM stock_features
        WHERE market = ? AND symbol = ?
          AND close IS NOT NULL
        ORDER BY row_num DESC
        LIMIT 250
    """
    try:
        with _db_connection() as con:
            df = con.execute(ohlcv_query, [market, proxy_symbol]).df()
    except Exception as e:
        logger.warning("レジーム判定用データ取得失敗: %s", e, exc_info=True)
        return "unknown"

    if df.empty:
        return "unknown"

    from src.market_data.market_regime import get_market_regime

    regimes = get_market_regime(df.sort_values("Date"))
    if regimes.empty:
        return "unknown"

    latest_regime = str(regimes.iloc[-1])
    logger.info("[allocator] 市場レジーム判定: market=%s regime=%s", market, latest_regime)
    return latest_regime


def _apply_correlation_penalty(
    symbols: list[str],
    weights: dict[str, float],
    returns: pd.DataFrame,
) -> dict[str, float]:
    """相関係数が高い銘柄ペアの配分を抑制する。

    相関が高い（> _CORR_PENALTY_THRESHOLD）ペアのうち、
    重みが小さい方の銘柄の重みを比例的に削減する。
    """
    if returns.empty or len(returns.columns) < 2:
        return weights

    # 共通銘柄のみ対象
    common = [s for s in symbols if s in returns.columns]
    if len(common) < 2:
        return weights

    corr = returns[common].corr()
    penalized = dict(weights)

    for i, sym_a in enumerate(common):
        for sym_b in common[i + 1 :]:
            raw = corr.loc[sym_a, sym_b] if sym_a in corr.index else 0.0
            corr_val = float(raw)  # type: ignore[arg-type]
            if abs(corr_val) <= _CORR_PENALTY_THRESHOLD:
                continue
            # 超過分を線形スケールでペナルティ（最大50%削減）
            excess = (abs(corr_val) - _CORR_PENALTY_THRESHOLD) / (1.0 - _CORR_PENALTY_THRESHOLD)
            penalty_factor = 1.0 - 0.5 * excess
            # 重みが小さい方を削減
            if penalized.get(sym_a, 0.0) <= penalized.get(sym_b, 0.0):
                penalized[sym_a] = penalized.get(sym_a, 0.0) * penalty_factor
            else:
                penalized[sym_b] = penalized.get(sym_b, 0.0) * penalty_factor

    return penalized


def compute_portfolio_allocation(
    candidates: list[dict],
    total_capital: float,
    market: str = "jp",
    regime_override: Optional[str] = None,
) -> list[AllocationResult]:
    """候補銘柄リストからポートフォリオ配分を計算する。

    Args:
        candidates: [{"market": ..., "symbol": ..., "diff_ratio": ...}, ...]
        total_capital: 総資本（円）
        market: マーケット識別子（相関計算とレジーム判定に使用）
        regime_override: 指定時はレジームをこの値で固定（テスト用）

    Returns:
        list[AllocationResult]: 配分結果（weight 降順でソート）
    """
    if not candidates:
        logger.warning("[allocator] 候補銘柄が空のため配分をスキップ")
        return []

    # --- 1. 正のシグナルのみ対象にし、上位 MAX_POSITIONS に絞る ---
    positive = [c for c in candidates if float(c.get("diff_ratio", 0.0)) > 0.0]
    if not positive:
        logger.info("[allocator] 正のシグナルを持つ候補銘柄がありません")
        return []

    positive.sort(key=lambda c: float(c["diff_ratio"]), reverse=True)
    selected = positive[:MAX_POSITIONS]

    symbols = [c["symbol"] for c in selected]
    logger.info(
        "[allocator] 候補銘柄数=%d / 配分対象=%d (MAX_POSITIONS=%d)",
        len(candidates),
        len(selected),
        MAX_POSITIONS,
    )

    # --- 2. 信頼度マップを取得 ---
    confidence_map = _load_confidence_map(market, symbols)
    for sym in symbols:
        if sym not in confidence_map:
            confidence_map[sym] = 0.5  # データなしは中立

    # --- 3. 相関計算用リターンを取得 ---
    returns_df = _load_returns_for_corr(symbols, market)

    # --- 4. レジーム判定 ---
    regime = _detect_regime(market, regime_override)
    regime_scale = _REGIME_SCALE.get(regime, 0.8)

    # --- 5. 基本スコア = diff_ratio × 信頼度 ---
    raw_scores: dict[str, float] = {}
    for c in selected:
        sym = c["symbol"]
        diff = max(0.0, float(c.get("diff_ratio", 0.0)))
        conf = confidence_map.get(sym, 0.5)
        raw_scores[sym] = diff * conf

    # --- 6. 相関ペナルティ適用 ---
    penalized_scores = _apply_correlation_penalty(symbols, raw_scores, returns_df)

    # --- 7. レジームスケール適用 ---
    scaled_scores: dict[str, float] = {
        sym: score * regime_scale for sym, score in penalized_scores.items()
    }

    # --- 8. 正規化（合計1.0）---
    total_score = sum(scaled_scores.values())
    if total_score <= 0.0:
        logger.warning("[allocator] スコア合計がゼロのため均等配分にフォールバック")
        n = len(selected)
        normalized: dict[str, float] = {sym: 1.0 / n for sym in symbols}
    else:
        normalized = {sym: score / total_score for sym, score in scaled_scores.items()}

    # --- 9. MAX_POSITION_RATE でキャップ ---
    capped: dict[str, float] = {sym: min(w, MAX_POSITION_RATE) for sym, w in normalized.items()}

    # --- 10. AllocationResult を生成し、セクター制約でフィルタ ---
    results: list[AllocationResult] = []
    for c in selected:
        sym = c["symbol"]
        w = capped.get(sym, 0.0)
        conf = confidence_map.get(sym, 0.5)
        diff = float(c.get("diff_ratio", 0.0))
        sector: Optional[str]
        try:
            sector = get_symbol_sector(market, sym)
        except Exception:
            sector = None

        reason_parts = [
            f"signal={diff:.4f}",
            f"confidence={conf:.2f}",
            f"regime={regime}(x{regime_scale:.1f})",
        ]
        if w < normalized.get(sym, w):
            reason_parts.append(f"capped@MAX_POSITION_RATE={MAX_POSITION_RATE:.0%}")

        results.append(
            AllocationResult(
                market=market,
                symbol=sym,
                weight=w,
                position_size_ratio=w / MAX_POSITION_RATE if MAX_POSITION_RATE > 0 else 0.0,
                signal_strength=diff,
                confidence=conf,
                regime=regime,
                sector=sector,
                allocation_reason=", ".join(reason_parts),
                allocated_capital=w * total_capital,
            )
        )

    # セクター制約（sector が None の銘柄はスキップ）
    if MAX_SECTOR_POSITIONS > 0:
        with_sector = [r for r in results if r.sector and not r.sector.startswith("UNKNOWN:")]
        without_sector = [r for r in results if not r.sector or r.sector.startswith("UNKNOWN:")]
        filtered_sector = filter_by_sector_cap(
            with_sector,
            MAX_SECTOR_POSITIONS,
            sector_getter=lambda r: r.sector or "",
        )
        results = filtered_sector + without_sector

    # weight 降順でソート
    results.sort(key=lambda r: r.weight, reverse=True)

    total_weight = sum(r.weight for r in results)
    logger.info(
        "[allocator] 配分完了: %d銘柄 total_weight=%.3f regime=%s",
        len(results),
        total_weight,
        regime,
    )
    return results
