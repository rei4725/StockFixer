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
