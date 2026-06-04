"""discord_formatters モジュールのユニットテスト"""

import unittest
from dataclasses import dataclass

import pandas as pd


@dataclass
class _PredStub:
    symbol: str
    current_price: float
    avg_pred_price: float
    diff_ratio: float


class TestSignalEmoji(unittest.TestCase):
    def test_buy_sell_neutral_and_none(self):
        from src.reporting.discord.discord_formatters import signal_emoji

        assert signal_emoji(0.01) == "⬆️"
        assert signal_emoji(-0.01) == "⬇️"
        assert signal_emoji(0.0) == "⏺️"
        assert signal_emoji(0.004) == "⏺️"  # しきい値内は中立
        assert signal_emoji(None) == "⏺️"


class TestFormatPrice(unittest.TestCase):
    def test_thousands_separator_and_precision(self):
        from src.reporting.discord.discord_formatters import format_price

        assert format_price(2500.0) == "2,500"
        assert format_price(8.456) == "8.46"
        assert format_price(0.123) == "0.123"

    def test_none_and_nan(self):
        from src.reporting.discord.discord_formatters import format_price

        assert format_price(None) == "—"
        assert format_price(float("nan")) == "—"


class TestBuildPredictionList(unittest.TestCase):
    def test_emoji_symbol_and_change(self):
        from src.reporting.discord.discord_formatters import build_prediction_list

        results = [
            _PredStub("7203", 2500, 2530, 0.012),
            _PredStub("9984", 8400, 8420, 0.0024),
            _PredStub("7974", 7200, 7150, -0.0069),
        ]
        lines = build_prediction_list(results)
        assert len(lines) == 3
        assert "⬆️" in lines[0] and "**7203**" in lines[0] and "+1.2%" in lines[0]
        assert "⏺️" in lines[1]  # しきい値内は中立
        assert "⬇️" in lines[2] and "-0.7%" in lines[2]

    def test_max_n_limits_rows(self):
        from src.reporting.discord.discord_formatters import build_prediction_list

        results = [_PredStub(str(i), 100, 101, 0.01) for i in range(20)]
        assert len(build_prediction_list(results, max_n=5)) == 5

    def test_empty(self):
        from src.reporting.discord.discord_formatters import build_prediction_list

        assert build_prediction_list([]) == []


class TestNormalizeMarketCode(unittest.TestCase):
    def test_lowercased_input(self):
        from src.reporting.discord.discord_formatters import normalize_market_code

        assert normalize_market_code("jp") == "JP"

    def test_trimmed_whitespace(self):
        from src.reporting.discord.discord_formatters import normalize_market_code

        assert normalize_market_code("  nasdaq  ") == "NASDAQ"


class TestGetMarketEmoji(unittest.TestCase):
    def test_jp_market(self):
        from src.reporting.discord.discord_formatters import get_market_emoji

        assert get_market_emoji("JP") == "🇯🇵"

    def test_nasdaq_market(self):
        from src.reporting.discord.discord_formatters import get_market_emoji

        assert get_market_emoji("nasdaq") == "🇺🇸"

    def test_unknown_market(self):
        from src.reporting.discord.discord_formatters import get_market_emoji

        assert get_market_emoji("UNKNOWN") == "🌐"


class TestConvertDfForDiscord(unittest.TestCase):
    def test_renames_columns(self):
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["7203"],
                "current_price": [2500.0],
                "avg_pred_price": [2550.0],
            }
        )
        result = convert_df_for_discord(df)
        assert "シンボル" in result.columns
        assert "現在値" in result.columns
        assert "予想終値" in result.columns

    def test_computes_diff_ratio_when_missing(self):
        """予想変化率カラムがない場合に自動計算されること"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["7203"],
                "current_price": [2500.0],
                "avg_pred_price": [2525.0],
            }
        )
        result = convert_df_for_discord(df)
        assert "予想変化率" in result.columns
        # 1% 変化
        assert "+1.0%" in str(result["予想変化率"].iloc[0]) or "1%" in str(
            result["予想変化率"].iloc[0]
        )

    def test_format_percent_negative(self):
        """マイナス変化率が正しく符号付きで出力されること"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["9984"],
                "current_price": [2000.0],
                "avg_pred_price": [1960.0],  # -2%
            }
        )
        result = convert_df_for_discord(df)
        change = result["予想変化率"].iloc[0]
        assert "-" in str(change)

    def test_truncates_price_values(self):
        """価格値が小数第3位で切り捨てられること"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "current_price": [150.9999],
                "avg_pred_price": [151.9999],
            }
        )
        result = convert_df_for_discord(df)
        # floor(150.9999 * 1000) / 1000 = 150.999
        assert float(result["現在値"].iloc[0]) == 150.999

    def test_interval_column_shown_when_present(self):
        """pred_lower_10 / pred_upper_90 がある場合に予想± 列が追加されること"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "current_price": [100.0],
                "avg_pred_price": [105.0],
                "diff_ratio": [0.05],
                "pred_lower_10": [103.0],
                "pred_upper_90": [107.0],
            }
        )
        result = convert_df_for_discord(df)
        assert "予想±" in result.columns

    def test_interval_format(self):
        """予想± 列が [+X.X%~+Y.Y%] 形式で出力されること"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "current_price": [100.0],
                "avg_pred_price": [105.0],
                "diff_ratio": [0.05],
                "pred_lower_10": [103.0],
                "pred_upper_90": [107.0],
            }
        )
        result = convert_df_for_discord(df)
        interval_text = str(result["予想±"].iloc[0])
        assert "[" in interval_text and "~" in interval_text and "]" in interval_text

    def test_interval_column_absent_when_not_present(self):
        """pred_lower_10 / pred_upper_90 がない場合は予想± 列が出力されないこと"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "current_price": [100.0],
                "avg_pred_price": [105.0],
            }
        )
        result = convert_df_for_discord(df)
        assert "予想±" not in result.columns

    def test_zero_current_price_falls_back_to_empty_ratio(self):
        """current_price=0 のとき 0除算をキャッチして予想変化率が '' になること"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["7203"],
                "current_price": [0.0],
                "avg_pred_price": [2550.0],
            }
        )
        result = convert_df_for_discord(df)
        assert result["予想変化率"].iloc[0] == ""

    def test_format_percent_with_non_numeric_value(self):
        """diff_ratio が非数値文字列のとき format_percent がそのまま返すこと"""
        from src.reporting.discord.discord_formatters import convert_df_for_discord

        df = pd.DataFrame(
            {
                "symbol": ["7203"],
                "current_price": [2500.0],
                "avg_pred_price": [2550.0],
                "diff_ratio": ["invalid"],
            }
        )
        result = convert_df_for_discord(df)
        assert result["予想変化率"].iloc[0] == "invalid"
