"""ポートフォリオシミュレーション（リバランス・ウェイト計算・セクター制約）。

Issue #511: 肥大化した portfolio.py を責務分割。
日次シミュレーション本体と、それが利用するウェイト/セクター/リバランス
ヘルパーを集約する。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import MAX_SECTOR_POSITIONS
from src.backtest.data_port import get_backtest_data_port
from src.utils.logger import get_logger
from src.utils.regime_weights import get_regime_sector_weight
from src.utils.sector_constraints import filter_by_sector_cap, get_symbol_sector

logger = get_logger(__name__)


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[Any]:
    """リバランス日のリストを返す。"""
    if freq == "daily":
        return list(index)
    elif freq == "weekly":
        # 各週の最初の取引日（月曜相当）
        week_groups = pd.Series(index, index=index).groupby(pd.Grouper(freq="W-MON"))
        return [g.iloc[0] for _, g in week_groups if len(g) > 0]
    elif freq == "monthly":
        month_groups = pd.Series(index, index=index).groupby(pd.Grouper(freq="MS"))
        return [g.iloc[0] for _, g in month_groups if len(g) > 0]
    else:
        raise ValueError(f"未対応のリバランス頻度: {freq}")


def _softmax_weights(scores: pd.Series) -> pd.Series:
    """正のスコアのみを対象に softmax でウェイトを計算する。"""
    valid = scores.dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return pd.Series(dtype=float)
    exp_s = np.exp(valid - valid.max())  # 数値安定化
    return exp_s / exp_s.sum()


def _apply_sector_rotation(scores: pd.Series, regime: str) -> pd.Series:
    """レジームに応じたセクターウェイト乗数をスコア Series に適用して返す。"""
    result = scores.copy()
    for sym in result.index:
        sector = _get_portfolio_symbol_sector(str(sym))
        result[sym] *= get_regime_sector_weight(regime, sector)
    return result


def _simulate_portfolio(
    score_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    rebalance_dates: list[Any],
    top_n: int,
    initial_cash: float,
    fee_rate: float,
    max_sector_positions: int,
    use_sector_rotation: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    ポートフォリオシミュレーションを実行する。

    毎リバランス日に Top-N 銘柄をスコア比例配分で保有し、
    日次の portfolio_value と equal_weight_value を記録する。
    """
    # セクターローテーション用に市場レジームを事前計算
    regime_series: Optional[pd.Series] = None
    if use_sector_rotation:
        proxy_df = _build_market_proxy_frame(close_matrix)
        if not proxy_df.empty:
            regime_series = get_backtest_data_port().get_market_regime(proxy_df)

    rebalance_set = set(str(d)[:10] for d in rebalance_dates)

    cash = initial_cash
    # {symbol: {"qty": int, "price": float}}
    holdings: dict[str, dict[str, Any]] = {}
    prev_symbols: set[str] = set()
    holdings_records: list[dict[str, Any]] = []

    equity_rows: list[dict[str, Any]] = []

    # 等分戦略の基準として全銘柄を均等保有したシリーズ（簡易近似）
    eq_cash = initial_cash
    eq_holdings: dict[str, dict[str, Any]] = {}

    for date in score_matrix.index:
        date_str = str(date)[:10]
        prices_today = close_matrix.loc[date].dropna()

        # ─ リバランス ─
        if date_str in rebalance_set:
            scores_today = score_matrix.loc[date].dropna()

            # セクターローテーション: レジームに応じてスコアを調整
            if use_sector_rotation and regime_series is not None:
                date_ts = pd.Timestamp(date_str)
                current_regime = "range"
                if date_ts >= regime_series.index.min():
                    raw = regime_series.asof(date_ts)
                    if not pd.isna(raw):
                        current_regime = str(raw)
                scores_today = _apply_sector_rotation(scores_today, current_regime)

            # 上位 top_n を選択
            top_candidates = _limit_portfolio_candidates_by_sector(
                scores_today.nlargest(top_n * 3),
                max_sector_positions=max_sector_positions,
            ).head(top_n)
            weights = _softmax_weights(top_candidates)

            if not weights.empty:
                new_symbols = set(weights.index)

                # ターンオーバー計算
                if prev_symbols:
                    stayed = prev_symbols & new_symbols
                    turnover = 1.0 - len(stayed) / max(len(prev_symbols), len(new_symbols))
                else:
                    turnover = 1.0

                # 既存保有を全売却
                for sym, pos in holdings.items():
                    if sym in prices_today.index:
                        proceeds = pos["qty"] * prices_today[sym] * (1 - fee_rate)
                        cash += proceeds

                holdings = {}

                # 新規銘柄を購入
                total_budget = cash
                for sym_key, w in weights.items():
                    sym = str(sym_key)
                    if sym not in prices_today.index:
                        continue
                    budget = total_budget * float(w)
                    price = prices_today[sym]
                    if price <= 0:
                        continue
                    qty = int(budget / (price * (1 + fee_rate)))
                    if qty > 0:
                        cost = qty * price * (1 + fee_rate)
                        cash -= cost
                        holdings[sym] = {"qty": qty, "price": price}

                        holdings_records.append(
                            {
                                "rebalance_date": date_str,
                                "symbol": sym,
                                "sector": _get_portfolio_symbol_sector(sym),
                                "weight": float(w),
                                "score": float(top_candidates.get(sym, 0)),
                                "price": price,
                                "qty": qty,
                                "turnover": turnover,
                            }
                        )

                # 等分戦略のリバランス（全有効銘柄に均等配分）
                valid_syms = [s for s in scores_today.index if s in prices_today.index]
                if valid_syms:
                    for sym, pos in eq_holdings.items():
                        if sym in prices_today.index:
                            eq_cash += pos["qty"] * prices_today[sym] * (1 - fee_rate)
                    eq_holdings = {}
                    eq_per = eq_cash / len(valid_syms)
                    for sym in valid_syms:
                        price = prices_today[sym]
                        if price <= 0:
                            continue
                        qty = int(eq_per / (price * (1 + fee_rate)))
                        if qty > 0:
                            eq_cash -= qty * price * (1 + fee_rate)
                            eq_holdings[sym] = {"qty": qty}

                prev_symbols = new_symbols

        # ─ 日次ポートフォリオ価値 ─
        pf_value = cash + sum(
            pos["qty"] * prices_today.get(sym, pos.get("price", 0)) for sym, pos in holdings.items()
        )
        eq_value = eq_cash + sum(
            pos["qty"] * prices_today.get(sym, 0) for sym, pos in eq_holdings.items()
        )

        equity_rows.append(
            {
                "date": date_str,
                "portfolio_value": round(pf_value, 2),
                "equal_weight_value": round(eq_value, 2),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, holdings_records


def _get_portfolio_symbol_sector(sym_key: str) -> str:
    market, symbol = _split_symbol_key(sym_key)
    return get_symbol_sector(market, symbol)


def _split_symbol_key(sym_key: str) -> tuple[str, str]:
    if "_" not in sym_key:
        return "jp", sym_key
    return tuple(sym_key.split("_", 1))  # type: ignore[return-value]


def _limit_portfolio_candidates_by_sector(
    top_candidates: pd.Series,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
) -> pd.Series:
    if top_candidates.empty or max_sector_positions <= 0:
        return top_candidates.copy()

    ordered_items = list(top_candidates.items())
    selected_items = filter_by_sector_cap(
        ordered_items,
        max_sector_positions=max_sector_positions,
        sector_getter=lambda item: _get_portfolio_symbol_sector(str(item[0])),
    )
    if not selected_items:
        return top_candidates.iloc[0:0].copy()
    return pd.Series(
        data=[float(score) for _, score in selected_items],
        index=[str(symbol) for symbol, _ in selected_items],
        dtype=float,
    )


def _build_market_proxy_frame(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """複数銘柄の終値行列から市場全体の簡易プロキシ価格を構築する。"""
    if close_matrix is None or close_matrix.empty:
        return pd.DataFrame()

    aligned = close_matrix.sort_index().ffill()
    proxy_df = pd.DataFrame(index=aligned.index)
    proxy_df["Close"] = aligned.mean(axis=1, skipna=True)
    proxy_df["High"] = aligned.max(axis=1, skipna=True)
    proxy_df["Low"] = aligned.min(axis=1, skipna=True)
    return proxy_df.dropna(subset=["Close"])
