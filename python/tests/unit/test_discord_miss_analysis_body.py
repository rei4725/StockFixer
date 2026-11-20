"""ユニットテスト: 週次レポートの外れ原因分析の知らせ send_miss_analysis_summary の本文（#2070）"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.prediction.miss_analysis import MissCauseFeature
from src.reporting.discord.discord_utils import send_miss_analysis_summary

_RATE_LIMITER = "src.reporting.discord.webhook_sender._rate_limiter.check_and_record"
_POST = "src.reporting.discord.webhook_sender._post_webhook"
_NAME_SUFFIX = "src.reporting.discord.notifications_drift.symbol_name_suffix"


def _embed_from_post(mock_post) -> dict:
    """_post_webhook モックから最初の embed を取り出す。"""
    payload = mock_post.call_args.kwargs["json_payload"]
    return payload["embeds"][0]


def _ok_response() -> MagicMock:
    return MagicMock(raise_for_status=MagicMock(return_value=None))


def _cause(feature: str, rank: int = 1, count: int = 2) -> MissCauseFeature:
    return MissCauseFeature(feature=feature, shap_rank=rank, shap_mean=0.1, miss_count=count)


def _miss_df(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["market", "symbol", "abs_error", "predicted_ratio", "actual_ratio"]
    )


@patch(_NAME_SUFFIX, return_value="")
@patch(_RATE_LIMITER, return_value=(True, None))
@patch(_POST)
class TestSendMissAnalysisSummary(unittest.TestCase):
    def test_empty_and_none_skip_sending(self, mock_post, mock_rl, _mock_name):
        """空の DataFrame と None なら送らずに True"""
        self.assertTrue(send_miss_analysis_summary(pd.DataFrame(), {}))
        self.assertTrue(send_miss_analysis_summary(None, {}))
        mock_post.assert_not_called()
        mock_rl.assert_not_called()

    def test_title_color_and_header(self, mock_post, _mock_rl, _mock_name):
        """題と色、見出しの行の期間"""
        mock_post.return_value = _ok_response()

        result = send_miss_analysis_summary(_miss_df([("jp", "7203", 0.05, 0.01, 0.06)]), {}, 14)

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "予測外れ原因分析")
        self.assertEqual(embed["color"], 0xFF8C00)
        first = embed["description"].split("\n")[0]
        self.assertTrue(first.startswith("**[予測外れ原因分析] "), first)
        self.assertTrue(first.endswith(" （直近14日）**"), first)

    def test_symbol_line_sign_and_format(self, mock_post, _mock_rl, _mock_name):
        """予測が負・実績が正の符号と小数2桁の %"""
        mock_post.return_value = _ok_response()

        send_miss_analysis_summary(_miss_df([("jp", "7203", 0.0456, -0.0123, 0.0333)]), {})

        lines = _embed_from_post(mock_post)["description"].split("\n")
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2], "● `jp/7203` 外れ幅=4.56%  予測=-1.23% / 実績=+3.33%")
        self.assertEqual(len(lines), 3)

    def test_symbol_name_suffix_follows_symbol(self, mock_post, _mock_rl, mock_name):
        """銘柄名の接尾辞は銘柄の後ろに付く"""
        mock_post.return_value = _ok_response()
        mock_name.return_value = " トヨタ"

        send_miss_analysis_summary(_miss_df([("jp", "7203", 0.01, 0.02, -0.01)]), {})

        lines = _embed_from_post(mock_post)["description"].split("\n")
        self.assertEqual(lines[2], "● `jp/7203` トヨタ 外れ幅=1.00%  予測=+2.00% / 実績=-1.00%")
        mock_name.assert_called_once_with("jp", "7203")

    def test_main_causes_first_three_only(self, mock_post, _mock_rl, _mock_name):
        """主要因は先頭3件だけ。要因の無い銘柄は主要因の行が無い"""
        mock_post.return_value = _ok_response()
        df = _miss_df([("jp", "7203", 0.05, 0.01, 0.06), ("us", "AAPL", 0.04, -0.02, 0.02)])
        causes = {
            ("jp", "7203"): [
                _cause("rsi_14", 1, 5),
                _cause("macd", 2, 4),
                _cause("volume_ratio", 3, 3),
                _cause("atr_14", 4, 2),
            ],
        }

        send_miss_analysis_summary(df, causes)

        lines = _embed_from_post(mock_post)["description"].split("\n")
        self.assertEqual(
            lines[3], "  主要因: rsi_14(rank#1,5回), macd(rank#2,4回), volume_ratio(rank#3,3回)"
        )
        self.assertNotIn("atr_14", lines[3])
        self.assertTrue(lines[4].startswith("● `us/AAPL`"), lines[4])
        self.assertEqual(len(lines), 5)
        self.assertEqual(sum(1 for line in lines if "主要因" in line), 1)

    def test_repeat_causes_three_or_more_sorted(self, mock_post, _mock_rl, _mock_name):
        """3銘柄以上の特徴量だけが銘柄数の多い順で繰り返し外れ要因に並ぶ"""
        mock_post.return_value = _ok_response()
        df = _miss_df([("jp", f"S{i}", 0.01, 0.01, 0.0) for i in range(4)])
        causes = {
            ("jp", "S0"): [_cause("macd"), _cause("rsi_14"), _cause("atr_14")],
            ("jp", "S1"): [_cause("macd"), _cause("rsi_14"), _cause("atr_14")],
            ("jp", "S2"): [_cause("rsi_14"), _cause("macd")],
            ("jp", "S3"): [_cause("rsi_14")],
        }

        send_miss_analysis_summary(df, causes)

        lines = _embed_from_post(mock_post)["description"].split("\n")
        self.assertEqual(lines[-2], "")
        self.assertEqual(lines[-1], "⚠️ 繰り返し外れ要因: rsi_14（4銘柄）, macd（3銘柄）")

    def test_no_repeat_line_when_two_or_fewer(self, mock_post, _mock_rl, _mock_name):
        """2銘柄以下にしか出ない特徴量だけなら繰り返し外れ要因の行が無い"""
        mock_post.return_value = _ok_response()
        df = _miss_df([("jp", f"S{i}", 0.01, 0.01, 0.0) for i in range(3)])
        causes = {
            ("jp", "S0"): [_cause("macd"), _cause("rsi_14")],
            ("jp", "S1"): [_cause("macd")],
            ("jp", "S2"): [_cause("rsi_14")],
        }

        send_miss_analysis_summary(df, causes)

        description = _embed_from_post(mock_post)["description"]
        self.assertNotIn("繰り返し外れ要因", description)


if __name__ == "__main__":
    unittest.main()
