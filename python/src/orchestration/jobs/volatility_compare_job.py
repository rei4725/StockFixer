"""週のジョブ: 値動きに合わせた量と損切りの線の比べを results/ の JSON に書く（#1547）。

発注の道には触れない。結果のファイルは、発注を変えるかを決める別の Issue の材料になる。
"""

from __future__ import annotations

import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_weekly_volatility_sizing_compare() -> None:
    """全体が落ちた回は投げ直し、run_job に error と記録させて失敗の知らせを出す。"""
    import pandas as pd

    from src.backtest.volatility_compare import (
        COMPARE_LOOKBACK_YEARS,
        run_volatility_sizing_compare,
        write_compare_result,
    )
    from src.utils.data_path_utils import get_results_dir
    from src.utils.japan_time import now_jst
    from src.watchlist.batch_runner import load_target_symbols

    logger.info("=== 週次 値動きの量と損切りの線の比べ開始 ===")
    try:
        # コンテナは UTC のため、日本時間の日付で期間と週を決める
        today = now_jst().date()
        start = pd.Timestamp(today) - pd.DateOffset(years=COMPARE_LOOKBACK_YEARS)
        result = run_volatility_sizing_compare(
            load_target_symbols(), start.date().isoformat(), today.isoformat()
        )
        path = write_compare_result(
            result,
            today,
            os.path.join(get_results_dir(), "backtest_compare", "volatility_sizing"),
        )
    except Exception as e:
        logger.error("週次 値動きの量と損切りの線の比べ失敗: %s", e, exc_info=True)
        raise
    logger.info("=== 週次 値動きの量と損切りの線の比べ完了: %s ===", path)
