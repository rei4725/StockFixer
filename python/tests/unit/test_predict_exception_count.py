"""日次の予想で銘柄ごとの例外を数え、完了のログと [3/6] の知らせに出すテスト（#1118）"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from src.prediction.types import PredictionResult

_DAILY = "src.orchestration.jobs.daily"
_DISCORD = "src.reporting.discord.discord_utils"
_SYMBOLS = [("jp", "5202"), ("jp", "7203"), ("jp", "9984"), ("jp", "6758")]


def _fake_predict(market, symbol, model_types=None, on_stale=None):
    """5202 は例外、9984 は特徴量が古い、6758 はそのほかの理由で予想なし。"""
    if symbol == "5202":
        raise RuntimeError("feature mismatch")
    if symbol == "9984":
        on_stale(market, symbol)
        return None
    if symbol == "6758":
        return None
    return MagicMock(spec=PredictionResult, market=market, symbol=symbol)


class TestPredictAllUnifiedExceptionCount:
    def _run(self, caplog, **kwargs):
        from src.prediction.prediction_pipeline import predict_all_unified

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "src.prediction.predict_unified.predict_with_unified_model",
                    side_effect=_fake_predict,
                )
            )
            stack.enter_context(patch("src.prediction.predict_unified.preload_models"))
            stack.enter_context(
                patch(
                    "src.prediction.prediction_pipeline.get_active_symbols", return_value=_SYMBOLS
                )
            )
            with caplog.at_level("INFO", logger="src.prediction"):
                return predict_all_unified(max_workers=2, **kwargs)

    def test_completion_log_separates_exception_count(self, caplog):
        """例外の数が特徴量が古いため予想なしと分けて完了のログに出て、戻り値は list のまま"""
        errors: list[tuple[str, str]] = []
        rows = self._run(caplog, on_error=lambda m, s: errors.append((m, s)))

        assert isinstance(rows, list)
        assert [r.symbol for r in rows] == ["7203"]
        assert errors == [("jp", "5202")]
        completion = [
            r.getMessage() for r in caplog.records if "予測完了（統合モデル）" in r.getMessage()
        ]
        assert len(completion) == 1
        assert "1/4銘柄" in completion[0]
        assert "特徴量が古いため予想なし: 1銘柄" in completion[0]
        assert "例外で予想なし: 1銘柄" in completion[0]
        assert "そのほかの理由で予想なし: 1銘柄" in completion[0]
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any("5202" in w for w in warnings)

    def test_without_callback_returns_rows_as_before(self, caplog):
        """コールバックを渡さない呼び方（今までの呼び方）は今までどおり動く"""
        rows = self._run(caplog)

        assert [r.symbol for r in rows] == ["7203"]


def _run_daily_pipeline(predict_side_effect):
    """日次の道を外の依存を差し替えて流し、差し替えた mock を返す。"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        mocks = {
            "webhook": p(f"{_DISCORD}.send_webhook_text"),
            "completion": p(f"{_DISCORD}.send_daily_pipeline_completion"),
            "error": p(f"{_DISCORD}.send_daily_pipeline_error"),
            "predict": p(
                "src.prediction.prediction_pipeline.predict_all_unified",
                side_effect=predict_side_effect,
            ),
            "output": p("src.prediction.prediction_pipeline.output_top_worst_results"),
        }
        p(f"{_DISCORD}.send_accuracy_summary")
        p("src.prediction.prediction_pipeline.run_accuracy_check")
        p("src.prediction.predict_unified.preload_models", return_value=[])
        p("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline")
        p(f"{_DAILY}._raise_if_fetch_mostly_unavailable")
        p(f"{_DAILY}.production_already_predicted_today", return_value=False)
        p(f"{_DAILY}.write_accuracy_json_without_sectors", return_value=None)
        p(f"{_DAILY}.append_accuracy_json_sectors")
        p(f"{_DAILY}.fetch_previous_run_stats", return_value=None)
        p(f"{_DAILY}.evaluate_output_invariants_stage", return_value=([], {}))
        p(f"{_DAILY}.run_daily_drift_check")
        p(f"{_DAILY}.record_health_degraded_streak")
        p(f"{_DAILY}.evaluate_and_notify_alerts")
        p(
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository"
        )

        run_daily_pipeline()
    return mocks


class TestDailyPredictExceptionNotice:
    def test_exceptions_send_one_notice_and_run_later_stages(self):
        """例外のあった回は知らせを1回送り、予想の保存と [5/6] の完了の知らせは呼ばれる"""
        rows = [MagicMock(spec=PredictionResult)]

        def predict(*_args, on_error=None, **_kwargs):
            on_error("jp", "5202")
            on_error("jp", "6758")
            return rows

        mocks = _run_daily_pipeline(predict)

        mocks["webhook"].assert_called_once()
        text = mocks["webhook"].call_args.args[0]
        assert "[3/6]" in text
        assert "2 件" in text
        assert "jp_5202" in text
        mocks["output"].assert_called_once()
        assert mocks["output"].call_args.args[0] is rows
        mocks["completion"].assert_called_once()
        mocks["error"].assert_not_called()

    def test_no_exceptions_send_no_notice(self):
        """例外が0件の回は新しい知らせを送らない"""
        mocks = _run_daily_pipeline(lambda *_a, **_k: [])

        mocks["webhook"].assert_not_called()
        mocks["output"].assert_called_once()
        mocks["completion"].assert_called_once()
