"""Unit tests: 紙の損益レポートと週次ルール評価の完了の知らせの本文 (#2048)

send_paper_trade_position_report と send_rule_evaluation_completion の
本文を組み立てる行を通す。送信の関数と銘柄名の引き当て（DB を読む）を
差し替え、渡った本文を今の形のまま確かめる。
"""

import unittest
from unittest.mock import patch

_MOD = "src.reporting.discord.discord_utils"
_NOW = "2026/11/20 15:30"


def _summary(**overrides) -> dict:
    base = {
        "realized_pnl": 30000.0,
        "unrealized_pnl": 20000.0,
        "total_pnl": 50000.0,
        "balance": 1030000.0,
        "initial_balance": 1000000.0,
        "trade_count": 12,
        "started_at": None,
    }
    base.update(overrides)
    return base


@patch(f"{_MOD}.format_jst", return_value=_NOW)
@patch(f"{_MOD}.symbol_name_suffix", return_value="")
@patch(f"{_MOD}.send_webhook_text_chunked", return_value=True)
class TestPaperTradePositionReport(unittest.TestCase):
    """紙の損益レポートの本文"""

    def _call(self, mock_send, positions, summary) -> str:
        from src.reporting.discord.discord_utils import send_paper_trade_position_report

        result = send_paper_trade_position_report(positions, summary)
        self.assertTrue(result)
        mock_send.assert_called_once()
        self.assertEqual(mock_send.call_args.kwargs, {"preserve_lines": False})
        return mock_send.call_args.args[0]

    def test_empty_positions_full_body(self, mock_send, _suffix, _now):
        """ポジションが空のときの本文全体（利回り +5.00%・運用開始日 -）"""
        body = self._call(mock_send, [], _summary())

        expected = "\n".join(
            [
                f"**[ペーパートレード損益レポート] {_NOW}**",
                "",
                "**ポジション一覧**",
                "現在保有中のポジションはありません。",
                "",
                "**通算損益サマリー**",
                "• 実現損益　　: `+30,000円`",
                "• 含み損益　　: `+20,000円`",
                "• 通算損益　　: `+50,000円` 📈  (+5.00%)",
                "• 現在残高　　: `1,030,000円`",
                "• 取引回数　　: `12回`",
                "• 運用開始日　: `-`",
            ]
        )
        self.assertEqual(body, expected)

    def test_position_rows_and_icons(self, mock_send, suffix, _now):
        """含み損益の符号でアイコンが分かれ、表の行が今の形で出ること"""
        positions = [
            {
                "symbol": "7203",
                "qty": 100,
                "avg_price": 2500.0,
                "current_price": 2600.0,
                "unrealized_pnl": 10000.0,
            },
            {
                "symbol": "AAPL",
                "qty": 5,
                "avg_price": 30000.0,
                "current_price": 29000.0,
                "unrealized_pnl": -5000.0,
            },
        ]
        body = self._call(mock_send, positions, _summary(unrealized_pnl=5000.0))

        lines = body.split("\n")
        header = "**銘柄         | 保有数 |  平均取得  |  現在値  |  含み損益**"
        idx = lines.index(header)
        self.assertEqual(lines[idx + 1], "─" * 56)
        self.assertEqual(
            lines[idx + 2],
            "`7203    ` |   100株 |    2,500円 |    2,600円 |    +10,000円 📈",
        )
        self.assertEqual(
            lines[idx + 3],
            "`AAPL    ` |     5株 |   30,000円 |   29,000円 |     -5,000円 📉",
        )
        self.assertNotIn("現在保有中のポジションはありません。", body)
        self.assertEqual([c.args for c in suffix.call_args_list], [(None, "7203"), (None, "AAPL")])

    def test_negative_total_icon_and_rate(self, mock_send, _suffix, _now):
        """通算損益が負なら 📉 と負の利回りが出ること"""
        body = self._call(mock_send, [], _summary(total_pnl=-25000.0, balance=975000.0))
        self.assertIn("• 通算損益　　: `-25,000円` 📉  (-2.50%)", body.split("\n"))
        self.assertIn("• 現在残高　　: `975,000円`", body.split("\n"))

    def test_zero_initial_balance_gives_zero_rate(self, mock_send, _suffix, _now):
        """initial_balance が 0 なら0割りせず利回りは +0.00% になること"""
        body = self._call(mock_send, [], _summary(initial_balance=0))
        self.assertIn("• 通算損益　　: `+50,000円` 📈  (+0.00%)", body.split("\n"))

    def test_started_at_none_shows_dash(self, mock_send, _suffix, _now):
        """started_at が None なら運用開始日は - になること"""
        body = self._call(mock_send, [], _summary(started_at=None))
        self.assertEqual(body.split("\n")[-1], "• 運用開始日　: `-`")

    def test_started_at_is_formatted_in_jst(self, mock_send, _suffix, _now):
        """started_at があれば JST の表示に変わり - にならないこと"""
        body = self._call(mock_send, [], _summary(started_at="2026-01-05T00:00:00+00:00"))
        last = body.split("\n")[-1]
        self.assertTrue(last.startswith("• 運用開始日　: `"))
        self.assertNotEqual(last, "• 運用開始日　: `-`")
        self.assertIn("2026", last)


@patch(f"{_MOD}.format_jst", return_value="2026/11/22 21:00 JST")
@patch(f"{_MOD}.send_webhook_text", return_value=True)
class TestRuleEvaluationCompletion(unittest.TestCase):
    """週次ルール評価の完了の知らせの本文"""

    _BASE = [
        "**[ルール評価完了] 2026/11/22 21:00 JST**",
        "マーケット: jp",
        "評価銘柄数: 120",
        "有効ルール発見: 45 銘柄",
        "スキップ (有効ルールなし): 75 銘柄",
    ]

    def test_no_errors_has_no_failure_line(self, mock_send, _now):
        """errors=0 では評価失敗の行が無いこと"""
        from src.reporting.discord.discord_utils import send_rule_evaluation_completion

        self.assertTrue(send_rule_evaluation_completion(120, 45, 75, "jp"))
        mock_send.assert_called_once()
        body = mock_send.call_args.args[0]
        self.assertEqual(body, "\n".join(self._BASE))
        self.assertNotIn("評価失敗", body)

    def test_errors_adds_failure_line_with_count(self, mock_send, _now):
        """errors>0 では件数つきの評価失敗の行が最後に足されること"""
        from src.reporting.discord.discord_utils import send_rule_evaluation_completion

        self.assertTrue(send_rule_evaluation_completion(120, 45, 75, "jp", errors=3))
        body = mock_send.call_args.args[0]
        expected = self._BASE + ["❌ 評価失敗: 3 銘柄（最優秀ルールの表が古いか欠けている）"]
        self.assertEqual(body, "\n".join(expected))

    def test_returns_sender_result(self, mock_send, _now):
        """送信の失敗は False のまま返ること"""
        from src.reporting.discord.discord_utils import send_rule_evaluation_completion

        mock_send.return_value = False
        self.assertFalse(send_rule_evaluation_completion(0, 0, 0, "us", errors=1))


if __name__ == "__main__":
    unittest.main()
