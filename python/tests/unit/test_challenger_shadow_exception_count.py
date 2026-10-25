"""日次の [3.5/6] challenger の影の予想で銘柄ごとの例外を数え、知らせに出すテスト（#1139）"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from src.prediction.types import PredictionResult

_DAILY = "src.orchestration.jobs.daily"
_DISCORD = "src.reporting.discord.discord_utils"
_SHADOW = "src.prediction.shadow_evaluation"
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


def _raise_always(market, symbol, model_types=None, on_stale=None):
    raise RuntimeError("feature mismatch")


def _run_daily_pipeline(caplog, predict_side_effect, model_files_exist=True):
    """日次の道を、challenger の予想の中身だけ本物にして流し、差し替えた mock を返す。"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        mocks = {
            "webhook": p(f"{_DISCORD}.send_webhook_text"),
            "completion": p(f"{_DISCORD}.send_daily_pipeline_completion"),
            "error": p(f"{_DISCORD}.send_daily_pipeline_error"),
            "output": p("src.prediction.prediction_pipeline.output_top_worst_results"),
            "drift": p(f"{_DAILY}.run_daily_drift_check"),
            "chal_predict": p(
                "src.prediction.predict_unified.predict_with_unified_model",
                side_effect=predict_side_effect,
            ),
        }
        p("src.prediction.prediction_pipeline.predict_all_unified", return_value=[])
        p(f"{_DISCORD}.send_accuracy_summary")
        p("src.prediction.prediction_pipeline.run_accuracy_check")
        p("src.prediction.predict_unified.preload_models", return_value=[])
        p("src.prediction.predict_unified.get_cached_model")
        p("src.utils.db.get_active_symbols", return_value=_SYMBOLS)
        p(
            f"{_SHADOW}.available_unified_challengers",
            return_value=["UnifiedStockXGBoost_challenger"] if model_files_exist else [],
        )
        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline")
        p(f"{_DAILY}._raise_if_fetch_mostly_unavailable")
        p(f"{_DAILY}.production_already_predicted_today", return_value=False)
        p(f"{_DAILY}.write_accuracy_json_without_sectors", return_value=None)
        p(f"{_DAILY}.append_accuracy_json_sectors")
        p(f"{_DAILY}.fetch_previous_run_stats", return_value=None)
        p(f"{_DAILY}.evaluate_output_invariants_stage", return_value=([], {}))
        p(f"{_DAILY}.record_health_degraded_streak")
        p(f"{_DAILY}.evaluate_and_notify_alerts")
        p(
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository"
        )

        with caplog.at_level("INFO"):
            run_daily_pipeline()
    return mocks


def _messages(caplog):
    return [r.getMessage() for r in caplog.records]


class TestChallengerShadowExceptionCount:
    def test_some_exceptions_logged_counted_notified_once_and_later_stages_run(self, caplog):
        """一部の銘柄の例外が完了のログに数で出て、[3.5/6] の知らせが1回出て、後の段が走る"""
        mocks = _run_daily_pipeline(caplog, _fake_predict)

        completion = [m for m in _messages(caplog) if "予測完了（challenger" in m]
        assert len(completion) == 1
        assert "1/4銘柄" in completion[0]
        assert "特徴量が古いため予想なし: 1銘柄" in completion[0]
        assert "例外で予想なし: 1銘柄" in completion[0]
        assert "そのほかの理由で予想なし: 1銘柄" in completion[0]

        mocks["webhook"].assert_called_once()
        text = mocks["webhook"].call_args.args[0]
        assert "[3.5/6]" in text
        assert "1 件" in text
        assert "jp_5202" in text
        challenger_outputs = [
            c
            for c in mocks["output"].call_args_list
            if c.kwargs.get("model_version") == "challenger"
        ]
        assert len(challenger_outputs) == 1
        assert [r.symbol for r in challenger_outputs[0].args[0]] == ["7203"]
        mocks["drift"].assert_called_once()
        mocks["completion"].assert_called_once()
        mocks["error"].assert_not_called()

    def test_all_exceptions_do_not_say_no_model(self, caplog):
        """challenger のファイルが在り全銘柄が例外のとき、モデルなしと出さずに知らせる"""
        mocks = _run_daily_pipeline(caplog, _raise_always)

        messages = _messages(caplog)
        assert not any("Challenger モデルなし（スキップ）" in m for m in messages)
        assert any("例外で予想なし: 4銘柄" in m for m in messages)
        mocks["webhook"].assert_called_once()
        assert "4 件" in mocks["webhook"].call_args.args[0]
        mocks["drift"].assert_called_once()
        mocks["completion"].assert_called_once()

    def test_no_model_files_skip_without_notice(self, caplog):
        """challenger のファイルが無いときは今のとおり知らせずに飛ばす"""
        mocks = _run_daily_pipeline(caplog, _fake_predict, model_files_exist=False)

        assert any("Challenger モデルなし（スキップ）" in m for m in _messages(caplog))
        mocks["chal_predict"].assert_not_called()
        mocks["webhook"].assert_not_called()
        mocks["drift"].assert_called_once()
        mocks["completion"].assert_called_once()
