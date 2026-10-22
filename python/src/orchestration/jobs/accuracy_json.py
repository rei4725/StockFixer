"""日次の採点の JSON（results/accuracy/YYYY-MM-DD.json）を書く段。

daily.py からの責務分離（ファイル行数ガード対策）。run_daily_pipeline() の
[2/6] で業種なしで書き、[6/6] の後で業種の内訳を書き足す（#843）。
"""

from typing import TYPE_CHECKING

from src.orchestration.jobs.common import _handle_stage_error
from src.orchestration.types import PipelineStage
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.prediction.accuracy_report import DailyAccuracyReport

logger = get_logger(__name__)


def production_already_predicted_today() -> bool:
    """prediction_results の production の回が既に今日のものなら True を返す（#904）。

    日次の道を頭からやり直すと（同じ回の中のやり直し・当日分の補完）、[3/6] が
    前日分を今日の予想で上書きした後に [2/6] が走り、まだ採点の無い今日の予想を
    読んで採点の JSON を0件で上書きする。そのときは採点と JSON を飛ばす。
    predicted_at は保存の側と同じく datetime.now() の YYYYMMDD_HHMMSS で比べる。
    読めなければ False を返し、今までどおり採点する。
    """
    from datetime import datetime

    try:
        from src.prediction.db import load_latest_prediction_timestamp

        latest = load_latest_prediction_timestamp(model_version="production")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[2/6] 前の回の日付の確認", e)
        return False
    if not latest or not str(latest).startswith(datetime.now().strftime("%Y%m%d")):
        return False
    logger.warning(
        "[2/6] production の予想が既に今日の回 (%s) のため、採点と採点の JSON を飛ばし"
        "前の回の JSON を残す（日次の道のやり直し）",
        latest,
    )
    return True


def write_accuracy_json_without_sectors() -> "DailyAccuracyReport | None":
    """採点の JSON を業種なしで書き、業種の書き足しに使う材料を返す（NON_CRITICAL）。

    3 の保存の前に呼ぶ（今日の採点の行を prediction_results のスナップショットで
    見分けるため）。業種の内訳は yfinance を約700銘柄ぶん引き [3/6] を数分遅らせる
    ので、ここでは引かない。失敗したら None を返す。
    """
    try:
        from src.prediction.accuracy_report import prepare_daily_accuracy_report

        accuracy_report = prepare_daily_accuracy_report()
        accuracy_report.write(sector_getter=None)
        logger.info("[2/6] 採点の JSON 書き出し完了（業種は道の最後に書き足す）")
        return accuracy_report
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[2/6] 採点の JSON 書き出し", e)
        return None


def append_accuracy_json_sectors(accuracy_report: "DailyAccuracyReport | None") -> None:
    """[2/6] で作った材料で、採点の JSON に業種の内訳を書き足す（NON_CRITICAL）。"""
    if accuracy_report is None:
        return
    try:
        accuracy_report.write_with_sectors()
        logger.info("採点の JSON の業種の書き足し完了")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "採点の JSON の業種の書き足し", e)
