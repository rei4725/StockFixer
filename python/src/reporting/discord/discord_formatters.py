"""Discord 表示向けの整形ユーティリティ。"""

from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_EMOJI = {
    "JP": "🇯🇵",
    "NASDAQ": "🇺🇸",
    "US": "🇺🇸",
}


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
            df["予想変化率"] = (df["予想終値"].astype(float) - df["現在値"].astype(float)) / df[
                "現在値"
            ].astype(float)
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
