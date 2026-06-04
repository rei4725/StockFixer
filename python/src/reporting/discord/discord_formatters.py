"""Discord 表示向けの整形ユーティリティ。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

MARKET_EMOJI = {
    "JP": "🇯🇵",
    "NASDAQ": "🇺🇸",
    "US": "🇺🇸",
}

# シグナル判定のしきい値（変化率）。determine_signal_label と揃える。
SIGNAL_BUY_THRESHOLD = 0.005
SIGNAL_SELL_THRESHOLD = -0.005


def signal_emoji(diff_ratio: float | None) -> str:
    """予想変化率からシグナル絵文字（⬆️ 買い / ⬇️ 売り / ⏺️ 中立）を返す。"""
    if diff_ratio is None:
        return "⏺️"
    if diff_ratio > SIGNAL_BUY_THRESHOLD:
        return "⬆️"
    if diff_ratio < SIGNAL_SELL_THRESHOLD:
        return "⬇️"
    return "⏺️"


def format_price(value: float | None) -> str:
    """価格を桁区切り付きで、値域に応じた桁数に整形する。"""
    if value is None or (isinstance(value, float) and value != value):  # None or NaN
        return "—"
    v = float(value)
    if abs(v) >= 100:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:,.2f}"
    return f"{v:.3f}"


def build_prediction_list(results: Sequence[Any], max_n: int = 10) -> list[str]:
    """予測結果リストを「絵文字＋1行/銘柄」のリスト表示行へ整形する。

    各 result は symbol / current_price / avg_pred_price / diff_ratio 属性を持つ
    PredictionResult を想定（duck typing）。スマホでの折り返しを避けるため
    1 銘柄 1 行に収める。

    例: `` 1`` ⬆️ **7203**  2,500 → 2,530  (+1.2%)
    """
    lines: list[str] = []
    for i, r in enumerate(list(results)[:max_n], start=1):
        ratio = getattr(r, "diff_ratio", None)
        emoji = signal_emoji(ratio)
        cur = format_price(getattr(r, "current_price", None))
        pred = format_price(getattr(r, "avg_pred_price", None))
        pct = f"{ratio * 100:+.1f}%" if ratio is not None else "—"
        lines.append(f"`{i:>2}` {emoji} **{getattr(r, 'symbol', '?')}**  {cur} → {pred}  ({pct})")
    return lines


def normalize_market_code(market: str) -> str:
    """市場コードをDiscord表示用の標準形へ正規化する。"""
    return market.strip().upper()


def get_market_emoji(market: str) -> str:
    """市場コードに対応する絵文字を返す。"""
    return MARKET_EMOJI.get(normalize_market_code(market), "🌐")


def convert_df_for_discord(df: pd.DataFrame) -> pd.DataFrame:
    """予測結果 DataFrame を Discord 向けに列名・書式を整える。"""
    columns_map = {
        "symbol": "シンボル",
        "current_price": "現在値",
        "avg_pred_price": "予想終値",
        "diff_ratio": "予想変化率",
        "予想値": "予想終値",
    }
    col_order = ["シンボル", "現在値", "予想終値", "予想変化率", "予想±"]
    df = df.rename(columns=columns_map)
    if "現在値" in df.columns and "予想終値" in df.columns and "予想変化率" not in df.columns:
        try:
            current = df["現在値"].astype(float)
            predicted = df["予想終値"].astype(float)
            ratio = (predicted - current) / current
            # 0除算で inf/-inf になる行は空文字に置換
            ratio = ratio.apply(
                lambda v: (
                    ""
                    if (
                        not isinstance(v, str)
                        and (v != v or v == float("inf") or v == float("-inf"))
                    )
                    else v
                )
            )
            df["予想変化率"] = ratio
        except (ValueError, TypeError, ZeroDivisionError):
            df["予想変化率"] = ""
    if "現在値" in df.columns:
        df["現在値"] = df["現在値"].apply(
            lambda value: np.floor(float(value) * 1000) / 1000 if pd.notnull(value) else value
        )
    if "予想終値" in df.columns:
        df["予想終値"] = df["予想終値"].apply(
            lambda value: np.floor(float(value) * 1000) / 1000 if pd.notnull(value) else value
        )
    if "予想変化率" in df.columns:

        def format_percent(value):
            try:
                ratio = float(value)
                sign = "+" if ratio >= 0 else ""
                return f"{sign}{ratio * 100:.2g}%"
            except (ValueError, TypeError):
                return value

        df["予想変化率"] = df["予想変化率"].apply(format_percent)

    if "pred_lower_10" in df.columns and "pred_upper_90" in df.columns and "現在値" in df.columns:

        def format_interval(row):
            try:
                lower = float(row["pred_lower_10"])
                upper = float(row["pred_upper_90"])
                current = float(row["現在値"])
                if current <= 0:
                    return ""
                lower_pct = (lower - current) / current * 100
                upper_pct = (upper - current) / current * 100
                return f"[{lower_pct:+.1f}%~{upper_pct:+.1f}%]"
            except (ValueError, TypeError, KeyError):
                return ""

        df["予想±"] = df.apply(format_interval, axis=1)

    return df[[column for column in col_order if column in df.columns]]
