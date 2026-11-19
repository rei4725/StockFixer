"""ユニットテスト: 相関リスク警告と戦略ファクトリーの完了・失敗の知らせの本文（#2040）"""

import unittest
from unittest.mock import MagicMock, patch

from src.reporting.discord.discord_utils import (
    send_correlation_alert,
    send_factory_completion,
    send_factory_failure,
)

_RATE_LIMITER = "src.reporting.discord.webhook_sender._rate_limiter.check_and_record"
_POST = "src.reporting.discord.webhook_sender._post_webhook"
_NAME_SUFFIX = "src.reporting.discord.notifications_drift.symbol_name_suffix"


def _embed_from_post(mock_post) -> dict:
    """_post_webhook モックから最初の embed を取り出す。"""
    payload = mock_post.call_args.kwargs["json_payload"]
    return payload["embeds"][0]


def _fields_map(embed: dict) -> dict:
    return {f["name"]: f["value"] for f in embed["fields"]}


def _ok_response() -> MagicMock:
    return MagicMock(raise_for_status=MagicMock(return_value=None))


@patch(_NAME_SUFFIX, return_value="")
@patch(_RATE_LIMITER, return_value=(True, None))
@patch(_POST)
class TestSendCorrelationAlert(unittest.TestCase):
    def test_title_color_and_number_format(self, mock_post, _mock_rl, _mock_name):
        """題・色と、ENC・閾値・平均相関の小数2桁の書式"""
        mock_post.return_value = _ok_response()

        result = send_correlation_alert(1.234, 2.5, 0.6789, 3, ["7203", "6758", "9984"])

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "相関リスク警告 — 分散度低下")
        self.assertEqual(embed["color"], 0xFF6600)
        lines = embed["description"].split("\n")
        self.assertTrue(lines[0].startswith("**[相関リスク警告] "), lines[0])
        self.assertEqual(lines[1], "ENC=1.23 < 閾値=2.50（新規エントリーをブロック）")
        self.assertEqual(lines[2], "平均相関係数: 0.68")
        self.assertEqual(lines[3], "保有銘柄数: 3")
        self.assertEqual(lines[4], "保有銘柄: `7203`, `6758`, `9984`")

    def test_only_first_ten_symbols_listed(self, mock_post, _mock_rl, _mock_name):
        """保有11件なら先頭10件だけ並ぶ"""
        mock_post.return_value = _ok_response()
        symbols = [f"S{i:02d}" for i in range(11)]

        send_correlation_alert(1.0, 2.0, 0.5, 11, symbols)

        description = _embed_from_post(mock_post)["description"]
        symbol_line = description.split("\n")[-1]
        expected = "保有銘柄: " + ", ".join(f"`S{i:02d}`" for i in range(10))
        self.assertEqual(symbol_line, expected)
        self.assertNotIn("`S10`", description)
        self.assertIn("保有銘柄数: 11", description)

    def test_empty_symbols_has_no_symbol_line(self, mock_post, _mock_rl, _mock_name):
        """空の保有では保有銘柄の行を出さない"""
        mock_post.return_value = _ok_response()

        send_correlation_alert(1.0, 2.0, 0.5, 0, [])

        lines = _embed_from_post(mock_post)["description"].split("\n")
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[3], "保有銘柄数: 0")
        self.assertFalse(any(line.startswith("保有銘柄: ") for line in lines))


@patch(_RATE_LIMITER, return_value=(True, None))
@patch(_POST)
class TestSendFactoryCompletion(unittest.TestCase):
    def test_with_passed_hypotheses(self, mock_post, _mock_rl):
        """合格1本以上は ✨ の題と青の色で、レポートの欄にファイル名を並べる"""
        mock_post.return_value = _ok_response()

        result = send_factory_completion(
            market="jp",
            evaluated=12,
            passed=2,
            champion_sharpe=1.23456,
            pbo=0.1,
            best_label="rsi_breakout",
            best_sharpe=1.5,
            report_hashes=["abc123", "def456"],
        )

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "🏭✨ 戦略ファクトリー夜間バッチ完了 (jp)")
        self.assertEqual(embed["color"], 0x00BFFF)
        fields = _fields_map(embed)
        self.assertEqual(fields["🧪 評価仮説"], "12 本")
        self.assertEqual(fields["✅ ゲート合格"], "2 本")
        self.assertEqual(fields["👑 チャンピオン Sharpe"], "1.235")
        self.assertEqual(fields["📉 バッチ PBO"], "0.100")
        self.assertEqual(fields["🥇 ベスト候補"], "rsi_breakout (Sharpe 1.500)")
        self.assertEqual(
            fields["📄 レポート (results/factory/reports/)"], "`abc123.json`\n`def456.json`"
        )

    def test_without_passed_hypotheses(self, mock_post, _mock_rl):
        """合格0本は ✨ の無い題と灰色で、レポートの欄は「なし」"""
        mock_post.return_value = _ok_response()

        send_factory_completion(
            market="us",
            evaluated=5,
            passed=0,
            champion_sharpe=0.8,
            pbo=0.55,
            best_label="ma_cross",
            best_sharpe=0.4,
            report_hashes=[],
        )

        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "🏭 戦略ファクトリー夜間バッチ完了 (us)")
        self.assertEqual(embed["color"], 0x808080)
        fields = _fields_map(embed)
        self.assertEqual(fields["✅ ゲート合格"], "0 本")
        self.assertEqual(fields["📄 レポート (results/factory/reports/)"], "なし")


@patch(_RATE_LIMITER, return_value=(True, None))
@patch(_POST)
class TestSendFactoryFailure(unittest.TestCase):
    def test_title_color_and_fields(self, mock_post, _mock_rl):
        """失敗の題・赤・市場・エラー・やり直しの案内"""
        mock_post.return_value = _ok_response()

        result = send_factory_failure("jp", "DB に届かない")

        self.assertTrue(result)
        embed = _embed_from_post(mock_post)
        self.assertEqual(embed["title"], "❌ 戦略ファクトリー夜間バッチ失敗")
        self.assertEqual(embed["color"], 0xFF0000)
        fields = _fields_map(embed)
        self.assertEqual(fields["🌐 市場"], "jp")
        self.assertEqual(fields["❌ エラー"], "DB に届かない")
        self.assertEqual(
            fields["🧪 評価"],
            "この回の評価は行われていません（run_strategy_factory.py でやり直せます）",
        )

    def test_long_error_is_truncated_to_1000_chars(self, mock_post, _mock_rl):
        """1000字を超えるエラーは先頭1000字に切る"""
        mock_post.return_value = _ok_response()
        error = "a" * 999 + "b" + "c" * 500

        send_factory_failure("us", error)

        value = _fields_map(_embed_from_post(mock_post))["❌ エラー"]
        self.assertEqual(len(value), 1000)
        self.assertEqual(value, "a" * 999 + "b")

    def test_empty_error_shows_unknown(self, mock_post, _mock_rl):
        """空のエラーは「不明」にする"""
        mock_post.return_value = _ok_response()

        send_factory_failure("us", "")

        self.assertEqual(_fields_map(_embed_from_post(mock_post))["❌ エラー"], "不明")


if __name__ == "__main__":
    unittest.main()
