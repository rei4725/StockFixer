"""
夜間の入れ替えで、発注する市場（jp）を十分に比べたかを線と比べる（#1693・#1699・#1701）。

銘柄の分母はウォッチリストの銘柄の数で、段差で stock_features から消えた銘柄の分も縮まない。
"""

from __future__ import annotations

import json
from typing import Optional

from src.prediction.backtest_promotion.judge import MIN_COVERAGE
from src.prediction.constant_prediction import CONSTANT_PREDICTION_STD, is_constant_prediction
from src.utils.data_path_utils import get_watchlist_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 発注する市場。ここを確かめられなかった回は勝ちにしない（#1675）
REQUIRED_MARKET = "jp"

_REQUIRED_NO_SYMBOLS = f"発注する市場 {REQUIRED_MARKET} の銘柄が無い"
_REQUIRED_ROWS_SHORT = (
    f"発注する市場 {REQUIRED_MARKET} の比べられた行が足りない"
    "（{n} 行、見込み {expected} 行、必要 {need:g} 以上）"
)
_REQUIRED_SYMBOLS_SHORT = (
    f"発注する市場 {REQUIRED_MARKET} の行のある銘柄が足りない"
    "（{n} 銘柄、全 {universe} 銘柄、必要 {need:g} 以上）"
)
_REQUIRED_DAYS_SHORT = (
    f"発注する市場 {REQUIRED_MARKET} の行のある営業日が足りない"
    "（{n} 日、東証の営業日 {business} 日、必要 {need:g} 以上）"
)
_REQUIRED_CONSTANT = (
    f"発注する市場 {REQUIRED_MARKET} の中で候補の予想が銘柄の間でほぼ同じ値"
    "（jp の日ごとの標準偏差 {std}、線 {line}）"
)


def required_market_constant_reason(markets: dict[str, dict]) -> Optional[str]:
    """発注する市場の中の日ごとの予想の標準偏差が線より小さいときの、勝ちにしない理由を返す。

    全市場を混ぜた日ごとの標準偏差（#1602）は、jp の中で同じ値でも us と値がずれれば線を
    越える（#1718）。jp の行が無い回は required_market_reasons が止め、2行以上の jp の日が
    無く標準偏差が出ない回（None）はばらつきを測れないので理由にしない。
    """
    summary = markets.get(REQUIRED_MARKET)
    if summary is None:
        return None
    std = summary["candidate"].get("prediction_std")
    if not is_constant_prediction(std):
        return None
    return _REQUIRED_CONSTANT.format(std=std, line=CONSTANT_PREDICTION_STD)


def load_watchlist_symbols(market: str) -> Optional[set[str]]:
    """config/watchlist.json のその市場の銘柄を返す。読めなければ None。

    load_target_symbols() と同じファイルを読む。prediction は src.watchlist を import できない
    （BC の独立の契約）ので、utils のパスから読む。
    """
    try:
        with open(get_watchlist_path(), encoding="utf-8") as f:
            watchlist = json.load(f)
    except (OSError, ValueError):
        logger.warning("ウォッチリストが読めない: market=%s", market, exc_info=True)
        return None
    symbols = watchlist.get(market) if isinstance(watchlist, dict) else None
    if not isinstance(symbols, list):
        logger.warning("ウォッチリストに銘柄の並びが無い: market=%s", market)
        return None
    return {str(symbol).upper() for symbol in symbols}


def required_market_reasons(markets: dict[str, dict]) -> list[str]:
    """発注する市場の行か銘柄か日が線に届かないときの、勝ちにしない理由を返す。

    銘柄の分母はウォッチリストの銘柄の数（読めない回は universe の銘柄の数）、日の分母は東証の
    営業日の数。行の見込みは行のある銘柄と日から数えるので、銘柄や日がごっそり欠けると一緒に
    縮み、行の線だけでは止まらない。universe も表の銘柄なので、段差で消えた銘柄の分だけ縮む。
    universe に発注する市場が無い回は、道の一致の検査（#1675）と同じく勝ちにしない。
    """
    summary = markets.get(REQUIRED_MARKET)
    if summary is None:
        return [_REQUIRED_NO_SYMBOLS]
    reasons: list[str] = []
    need_rows = summary["expected_rows"] * MIN_COVERAGE
    if summary["n_compared"] < need_rows:
        reasons.append(
            _REQUIRED_ROWS_SHORT.format(
                n=summary["n_compared"], expected=summary["expected_rows"], need=need_rows
            )
        )
    n_symbols, n_listed = summary["n_symbols"], summary["n_universe_symbols"]
    if summary["n_watchlist_symbols"] is not None:
        n_symbols = summary["n_watchlist_symbols_with_rows"]
        n_listed = summary["n_watchlist_symbols"]
    need_symbols = n_listed * MIN_COVERAGE
    if n_symbols < need_symbols:
        reasons.append(
            _REQUIRED_SYMBOLS_SHORT.format(n=n_symbols, universe=n_listed, need=need_symbols)
        )
    need_days = summary["n_business_dates"] * MIN_COVERAGE
    if summary["n_business_dates_with_rows"] < need_days:
        reasons.append(
            _REQUIRED_DAYS_SHORT.format(
                n=summary["n_business_dates_with_rows"],
                business=summary["n_business_dates"],
                need=need_days,
            )
        )
    return reasons
