"""ルールシグナルと価格予想の採点を日次・週次の道につなぐ段（NON_CRITICAL）。

どの段も例外を外に出さない。失敗しても本体の道を止めない。
daily.py / weekly.py からの責務分離（ファイル行数ガード対策）。
"""

from datetime import date

from src.orchestration.jobs.common import _handle_stage_error
from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

logger = get_logger(__name__)

SNAPSHOT_STAGE = "[3.6/6] 予想の記録"
DAILY_SCORING_STAGE = "[7/6] ルールシグナルの採点"
VERSIONS_STAGE = "ルールシグナルの版の記録"
WEEKLY_STAGE = "週次のルール採点"


def snapshot_predictions_stage() -> None:
    """prediction_results の今の行を履歴へ写す（版を添える）。"""
    try:
        from src.scoring.db import snapshot_predictions
        from src.scoring.version import read_app_version

        added = snapshot_predictions(read_app_version())
        logger.info("%s 完了: %s 行", SNAPSHOT_STAGE, added)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, SNAPSHOT_STAGE, e)


def daily_scoring_stage() -> None:
    """ルールシグナルを採点し、日次の JSON を書く。"""
    try:
        from src.scoring.rule_report import run_daily_scoring

        path = run_daily_scoring()
        logger.info("%s 完了: %s", DAILY_SCORING_STAGE, path)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, DAILY_SCORING_STAGE, e)


def log_rule_signal_versions_stage(signal_date: date, market: str) -> None:
    """その日のシグナル行に版を結びつける。"""
    try:
        from src.scoring.db import log_rule_signal_versions
        from src.scoring.version import read_app_version

        added = log_rule_signal_versions(signal_date, market, read_app_version())
        logger.info("%s 完了: %s 行", VERSIONS_STAGE, added)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, VERSIONS_STAGE, e)


def weekly_rule_scoring_stage(market: str, evaluated_on: date) -> None:
    """ルールの選び方の出入りを記録し、週の JSON を書く（評価ジョブの成否は変えない）。"""
    try:
        from src.scoring.db import snapshot_rule_best
        from src.scoring.rule_report import write_weekly_rule_accuracy

        snapshot_rule_best(evaluated_on, market)
        iso = evaluated_on.isocalendar()
        path = write_weekly_rule_accuracy(market, iso[0], iso[1])
        logger.info("%s 完了: %s", WEEKLY_STAGE, path)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, WEEKLY_STAGE, e)
