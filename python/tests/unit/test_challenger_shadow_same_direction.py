"""日次の [3.5/6] challenger の影の予想で、市場ごとの同じ向きの偏りをログに出すテスト（#2080）"""

from contextlib import ExitStack
from unittest.mock import patch

from src.prediction.types import PredictionResult

_SHADOW = "src.prediction.shadow_evaluation"
_WARN_PREFIX = "[3.5/6] Challenger の予想が同じ向きに偏った"


def _row(market: str, symbol: str, diff_ratio) -> PredictionResult:
    return PredictionResult(
        market=market,
        symbol=symbol,
        current_price=100.0,
        avg_pred_price=100.0,
        diff_ratio=diff_ratio,
        model_count=2,
    )


def _run_stage(caplog, rows):
    from src.orchestration.jobs.challenger_shadow import run_challenger_shadow_stage

    with ExitStack() as stack:
        output = stack.enter_context(
            patch("src.prediction.prediction_pipeline.output_top_worst_results")
        )
        stack.enter_context(patch(f"{_SHADOW}.predict_with_challenger_unified", return_value=rows))
        stack.enter_context(patch(f"{_SHADOW}.available_unified_challengers", return_value=[]))
        notify = stack.enter_context(
            patch("src.orchestration.jobs.challenger_shadow.notify_prediction_errors")
        )
        with caplog.at_level("INFO"):
            run_challenger_shadow_stage()
    return output, notify


def _warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "WARNING" and _WARN_PREFIX in r.getMessage()
    ]


def _direction_infos(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelname == "INFO" and "Challenger の予想の向き" in r.getMessage()
    ]


class TestChallengerShadowSameDirection:
    def test_jp_mostly_up_logs_warning(self, caplog):
        """jp に 20 銘柄・19 が上げ（95%）の予想で WARNING が1行出る"""
        rows = [_row("jp", f"{i:04d}", 0.01) for i in range(19)] + [_row("jp", "9999", -0.01)]
        output, _ = _run_stage(caplog, rows)

        warnings = _warnings(caplog)
        assert warnings == [f"{_WARN_PREFIX}: jp 19/20 銘柄が上げ"]
        assert _direction_infos(caplog) == []
        output.assert_called_once()

    def test_each_biased_market_logs_one_warning(self, caplog):
        """偏った市場ごとに WARNING が1行ずつ出る"""
        rows = [_row("jp", f"{i:04d}", 0.01) for i in range(20)] + [
            _row("us", f"S{i}", -0.02) for i in range(25)
        ]
        _run_stage(caplog, rows)

        assert _warnings(caplog) == [
            f"{_WARN_PREFIX}: jp 20/20 銘柄が上げ",
            f"{_WARN_PREFIX}: us 25/25 銘柄が下げ",
        ]

    def test_balanced_prediction_logs_info_not_warning(self, caplog):
        """偏らない予想では WARNING が出ず、市場ごとの上げ・下げ・総数の INFO が1行出る"""
        rows = (
            [_row("jp", f"{i:04d}", 0.01) for i in range(12)]
            + [_row("jp", f"D{i}", -0.01) for i in range(8)]
            + [_row("us", "AAPL", 0.02), _row("us", "MSFT", 0.0)]
        )
        output, _ = _run_stage(caplog, rows)

        assert _warnings(caplog) == []
        infos = _direction_infos(caplog)
        assert len(infos) == 1
        assert "jp 上げ 12・下げ 8・総数 20" in infos[0]
        assert "us 上げ 1・下げ 0・総数 2" in infos[0]
        output.assert_called_once()

    def test_check_failure_does_not_stop_stage(self, caplog):
        """判定が例外を投げても段は止まらず、保存と知らせが今のとおり呼ばれる"""
        rows = [_row("jp", f"{i:04d}", "not-a-number") for i in range(20)]
        output, notify = _run_stage(caplog, rows)

        output.assert_called_once()
        assert output.call_args.args[0] == rows
        assert output.call_args.kwargs == {
            "mode": "unified",
            "shadow_mode": True,
            "model_version": "challenger",
        }
        notify.assert_called_once()
        messages = [r.getMessage() for r in caplog.records]
        assert any("同じ向きの判定に失敗" in m for m in messages)
        assert any("完了: 20 銘柄" in m for m in messages)
