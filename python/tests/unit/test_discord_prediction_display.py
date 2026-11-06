"""ユニットテスト: Discord の予想の値の隠し方と、ルールのシグナルの実績の表示"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from config.feature_flags import flags
from src.orchestration.jobs.rule_signal_notify import notify_rule_signals
from src.reporting.discord.discord_bot import (
    build_prediction_list_message,
    build_prediction_table_text,
    build_signal_lines,
)
from src.reporting.discord.discord_formatters import (
    PREDICTED_CHANGE_HIDDEN_NOTE,
    build_prediction_list,
    hide_predicted_columns,
)
from src.reporting.discord.discord_utils import _send_prediction_snapshot, send_rule_daily_signals
from src.scoring.rule_report import load_rule_track_record


def _result(symbol="7203", ratio=0.0612):
    return SimpleNamespace(
        symbol=symbol, market="jp", current_price=2500, avg_pred_price=2653, diff_ratio=ratio
    )


def _signal(rule="bollinger_band", signal=1, symbol="7203"):
    return {"symbol": symbol, "rule": rule, "signal": signal, "price": 2500, "win_rate": 0.77}


def _stats(n=341, edge=0.1044, mean_return=0.0229):
    return {"n": n, "edge": edge, "mean_signed_return": mean_return}


def _record(rule="bollinger_band", direction="buy", **stats):
    return {rule: {direction: {"5": _stats(**stats)}}}


def _sent_rule_text(signals, track_record=None):
    with (
        patch("src.reporting.discord.discord_utils.send_webhook_text", return_value=True) as sent,
        patch("src.reporting.discord.discord_utils.symbol_name_suffix", return_value=""),
    ):
        send_rule_daily_signals(signals, "jp", 1, 0, 100, track_record=track_record)
    return sent.call_args.args[0]


@pytest.fixture(autouse=True)
def _no_symbol_names():
    with patch("src.reporting.discord.discord_formatters.symbol_name_suffix", return_value=""):
        yield


class TestHiddenByDefault:
    def test_default_flag_is_off(self):
        assert type(flags).model_fields["SHOW_PREDICTED_CHANGE"].default is False

    def test_list_shows_order_and_price_only(self):
        with patch.object(flags, "SHOW_PREDICTED_CHANGE", False):
            line = build_prediction_list([_result()])[0]
        assert line == "` 1` **7203**  2,500"

    def test_channel_message_says_why(self):
        with patch.object(flags, "SHOW_PREDICTED_CHANGE", False):
            message = build_prediction_list_message("jp", "上位", [_result()])
        assert PREDICTED_CHANGE_HIDDEN_NOTE in message
        assert "+6.1%" not in message

    def test_daily_snapshot_has_no_predicted_values(self):
        snapshot = SimpleNamespace(market="jp", top_results=[_result()], worst_results=[_result()])
        with (
            patch.object(flags, "SHOW_PREDICTED_CHANGE", False),
            patch("src.reporting.discord.discord_utils.send_webhook_notification") as sent,
        ):
            _send_prediction_snapshot(snapshot, "2026-11-06 07:00")
        body = sent.call_args.args[1]
        assert "+6.1%" not in body and "2,653" not in body and "予想上昇" not in body
        assert PREDICTED_CHANGE_HIDDEN_NOTE in body

    def test_signal_lines_drop_prediction_and_signal(self):
        with (
            patch.object(flags, "SHOW_PREDICTED_CHANGE", False),
            patch("src.reporting.discord.discord_bot.symbol_label", return_value="7203"),
        ):
            message = "\n".join(build_signal_lines("jp", "7203", 100.0, 106.0, 0.06, 3))
        assert "現在値" in message and "使用モデル" in message
        assert "予想終値" not in message and "予想変化率" not in message
        assert "シグナル" not in message

    def test_table_columns_are_dropped(self):
        df = pd.DataFrame(
            {"シンボル": ["7203"], "現在値": [1.0], "予想終値": [2.0], "予想変化率": ["+6%"]}
        )
        with patch.object(flags, "SHOW_PREDICTED_CHANGE", False):
            assert list(hide_predicted_columns(df).columns) == ["シンボル", "現在値"]

    def test_empty_results_stay_empty(self):
        with patch.object(flags, "SHOW_PREDICTED_CHANGE", False):
            assert build_prediction_table_text([]) == ""
            assert build_prediction_list_message("jp", "上位", []) == ""


class TestFlagOnRestoresValues:
    def test_list_shows_values(self):
        with patch.object(flags, "SHOW_PREDICTED_CHANGE", True):
            line = build_prediction_list([_result()])[0]
        assert "2,500 → 2,653" in line and "+6.1%" in line and "⬆️" in line

    def test_no_note_and_columns_kept(self):
        df = pd.DataFrame({"シンボル": ["7203"], "予想変化率": ["+6%"]})
        with patch.object(flags, "SHOW_PREDICTED_CHANGE", True):
            message = build_prediction_list_message("jp", "上位", [_result()])
            kept = hide_predicted_columns(df)
        assert PREDICTED_CHANGE_HIDDEN_NOTE not in message
        assert "予想変化率" in kept.columns


class TestRuleTrackRecordLine:
    def test_shows_live_edge_before_backtest(self):
        text = _sent_rule_text([_signal()], _record())
        assert "実績+10.4pt 平均+2.3% n=341" in text
        assert text.index("実績+10.4pt") < text.index("BT勝率=77.0%")

    def test_buy_and_sell_use_their_own_record(self):
        record = {"r": {"buy": {"5": _stats(edge=0.1)}, "sell": {"5": _stats(edge=-0.05)}}}
        text = _sent_rule_text([_signal("r", 1, "1111"), _signal("r", -1, "2222")], record)
        assert "実績+10.0pt" in text and "実績-5.0pt" in text

    def test_small_sample_shows_no_edge(self):
        text = _sent_rule_text([_signal()], _record(n=29))
        assert "実績不足 n=29" in text and "pt 平均" not in text

    def test_30_samples_is_enough(self):
        assert "実績+10.4pt" in _sent_rule_text([_signal()], _record(n=30))

    def test_missing_record(self):
        assert "実績なし" in _sent_rule_text([_signal()], {})
        assert "実績なし" in _sent_rule_text([_signal()], None)
        assert "実績なし" in _sent_rule_text([_signal()], _record(n=0, edge=None, mean_return=None))

    def test_missing_mean_return_is_left_out(self):
        text = _sent_rule_text([_signal()], _record(mean_return=None))
        assert "実績+10.4pt n=341" in text

    def test_legend_only_with_signals(self):
        assert "BT: バックテスト" in _sent_rule_text([_signal()], _record())
        assert "BT: バックテスト" not in _sent_rule_text([], _record())


class TestLoadRuleTrackRecord:
    @staticmethod
    def _write(directory, name, market="jp", rules=None):
        report = {"market": market, "windows": {"all": {"rules": rules or {}}}}
        (directory / name).write_text(json.dumps(report), encoding="utf-8")

    def test_reads_latest_file(self, tmp_path):
        self._write(tmp_path, "2026-11-05.json", rules={"old": {}})
        self._write(tmp_path, "2026-11-06.json", rules={"new": {}})
        assert load_rule_track_record("jp", tmp_path) == {"new": {}}

    def test_ignores_weekly_and_other_files(self, tmp_path):
        self._write(tmp_path, "2026-11-05.json", rules={"daily": {}})
        self._write(tmp_path, "2026-W45.json", rules={"weekly": {}})
        assert load_rule_track_record("jp", tmp_path) == {"daily": {}}

    def test_other_market_is_empty(self, tmp_path):
        self._write(tmp_path, "2026-11-06.json", market="us", rules={"x": {}})
        assert load_rule_track_record("jp", tmp_path) == {}

    def test_no_file_is_empty(self, tmp_path):
        assert load_rule_track_record("jp", tmp_path) == {}

    def test_broken_file_is_empty(self, tmp_path):
        (tmp_path / "2026-11-06.json").write_text("{", encoding="utf-8")
        assert load_rule_track_record("jp", tmp_path) == {}


class TestNotifyRuleSignals:
    STATS = {"buy_orders": 1, "sell_orders": 0}

    def test_passes_track_record_to_the_notice(self):
        with (
            patch("src.scoring.rule_report.load_rule_track_record", return_value={"r": {}}),
            patch("src.reporting.discord.discord_utils.send_rule_daily_signals") as sent,
        ):
            notify_rule_signals([], "jp", self.STATS, 100)
        assert sent.call_args.kwargs["track_record"] == {"r": {}}
        assert sent.call_args.kwargs["total"] == 100

    def test_notice_failure_does_not_stop_the_job(self):
        with patch(
            "src.reporting.discord.discord_utils.send_rule_daily_signals",
            side_effect=RuntimeError("webhook"),
        ):
            notify_rule_signals([], "jp", self.STATS, 100)
