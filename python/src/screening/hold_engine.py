"""保有/撤退エンジン本体

エントリー後のポジションを「道中の含み損に耐えて勝ち馬を数年走らせる」ための
ルールエンジン。バックテスト(#431)と実運用アラート(#434)の両方から呼べる
純粋関数（副作用なし）として実装する。依存は utils ＋ pandas のみ。

判定はそれぞれ小さな private 関数に分け、HoldRules で挙動を注入できるようにして
ある（撤退・利確の条件を後で差し替え可能にするため）。
"""

from __future__ import annotations

import pandas as pd

from src.screening.types import HoldRules, PositionEvent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _ma_window(rules: HoldRules) -> int:
    """トレーリング MA の窓幅（営業日）。40週=200営業日。"""
    return rules.trail_ma_weeks * 5


def _rolling_ma(close: pd.Series, rules: HoldRules) -> pd.Series:
    """長期移動平均（窓が満たないうちは NaN＝判定しない）。"""
    return close.rolling(_ma_window(rules)).mean()


def _is_ma_break(close_val: float, ma_val: float) -> bool:
    """長期トレンド破壊（= 長期 MA 割れ）。

    Stage 1 では「テーゼ撤退(thesis_break)」をこの MA 割れで代理する。
    財務悪化による thesis_break は Stage 2(#434) で拡張する。
    """
    return bool(not pd.isna(ma_val) and close_val < ma_val)


def _is_trail_stop(close_val: float, peak: float, rules: HoldRules) -> bool:
    """走行中の最高値からトレーリング幅だけ下落したか。"""
    return bool(close_val <= peak * (1.0 - rules.trail_stop_pct))


def _newly_crossed_multiples(
    multiple: float, achieved: set[float], rules: HoldRules
) -> list[float]:
    """この日に初めて到達した未達成の利確倍率（昇順）。"""
    if not rules.scale_out_multiples or rules.scale_out_fraction <= 0:
        return []
    return sorted(
        level for level in rules.scale_out_multiples if level not in achieved and multiple >= level
    )


def _scale_reason(level: float) -> str:
    """利確倍率に対応する reason（例: 2.0 -> "scale_2x"）。"""
    return f"scale_{int(level)}x"


def simulate_position(
    prices: pd.DataFrame,
    entry_date: str,
    rules: HoldRules = HoldRules(),
) -> list[PositionEvent]:
    """エントリー後のポジションを日次ループでシミュレートする。

    Args:
        prices: columns に date, Close を持つ昇順系列（エントリー日以降）。
        entry_date: エントリー日（YYYY-MM-DD）。prices に含まれること。
        rules: 保有/撤退ルール。

    Returns:
        発生した PositionEvent のリスト（entry → scale_out... → exit）。
        held_fraction が変化したイベント（entry/scale_out/exit）のみ記録する。

    Note:
        未来データを参照しない（ルックアヘッド禁止）。各営業日の判定は、
        その日までの終値のみから計算する（rolling MA も trailing）。
    """
    if prices is None or prices.empty:
        logger.warning("simulate_position: prices が空です")
        return []
    if "Close" not in prices.columns or "date" not in prices.columns:
        logger.warning("simulate_position: prices に date / Close 列がありません")
        return []

    df = prices.reset_index(drop=True)
    dates = df["date"].astype(str)
    close = df["Close"].astype(float)

    entry_idx_matches = df.index[dates == entry_date].tolist()
    if not entry_idx_matches:
        logger.warning("simulate_position: entry_date=%s が prices に存在しません", entry_date)
        return []
    entry_idx = int(entry_idx_matches[0])

    entry_price = float(close.iloc[entry_idx])
    if entry_price <= 0:
        logger.warning("simulate_position: entry_price=%s が不正です", entry_price)
        return []

    ma = _rolling_ma(close, rules)

    events: list[PositionEvent] = [
        PositionEvent(
            date=str(dates.iloc[entry_idx]),
            action="entry",
            price=entry_price,
            held_fraction=1.0,
            reason="entry",
            multiple=1.0,
        )
    ]

    peak = entry_price
    held_fraction = 1.0
    achieved: set[float] = set()

    for i in range(entry_idx + 1, len(df)):
        close_val = float(close.iloc[i])
        date_str = str(dates.iloc[i])
        peak = max(peak, close_val)
        multiple = close_val / entry_price

        # 1) 長期トレンド破壊（MA 割れ）→ 撤退
        if _is_ma_break(close_val, float(ma.iloc[i])):
            events.append(PositionEvent(date_str, "exit", close_val, 0.0, "ma_break", multiple))
            return events

        # 2) トレーリングストップ（高値からの下落）→ 撤退
        if _is_trail_stop(close_val, peak, rules):
            events.append(PositionEvent(date_str, "exit", close_val, 0.0, "trail_stop", multiple))
            return events

        # 3) 段階利確（未達成の倍率を初めて超えた日に部分売却）
        for level in _newly_crossed_multiples(multiple, achieved, rules):
            achieved.add(level)
            held_fraction = max(0.0, held_fraction - rules.scale_out_fraction)
            events.append(
                PositionEvent(
                    date_str,
                    "scale_out",
                    close_val,
                    held_fraction,
                    _scale_reason(level),
                    multiple,
                )
            )

    # データ末尾まで撤退しなければ最終日に手仕舞い
    last_idx = len(df) - 1
    last_close = float(close.iloc[last_idx])
    events.append(
        PositionEvent(
            date=str(dates.iloc[last_idx]),
            action="exit",
            price=last_close,
            held_fraction=0.0,
            reason="end_of_data",
            multiple=last_close / entry_price,
        )
    )
    return events
