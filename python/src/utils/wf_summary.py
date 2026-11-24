"""最新の Walk-Forward サマリー CSV の読み込み（昇格ゲートと月次レポートが共に使う）。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_latest_wf_summary(
    report_dir: Path,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """report_dir の最新の Walk-Forward サマリー CSV を (DataFrame, ファイル名) で返す。

    名前の逆順の先頭だけを読む。CSV が無いとき・読めないときは (None, None) を返す。
    """
    candidates = sorted(report_dir.glob("wf_summary_*.csv"), reverse=True)
    if not candidates:
        logger.warning("Walk-Forward レポート CSV が見つかりません")
        return None, None
    path = candidates[0]
    try:
        df = pd.read_csv(path)
        logger.info(f"WF サマリー読み込み: {path.name} ({len(df)} 銘柄)")
        return df, path.name
    except Exception as e:
        logger.error(f"WF サマリー読み込み失敗: {e}", exc_info=True)
        return None, None
