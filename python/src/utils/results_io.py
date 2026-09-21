"""結果 CSV の保存を一本化する（Phase 1 / PR-2）。

`results/` 配下への保存は「ensure_dir → タイムスタンプ → to_csv → パス返却」という
同じ骨格が 5 箇所（portfolio / longterm / optimizer / stress_test / trend_screener）
に散っており、タイムゾーンが UTC とローカル時刻で割れていた。本 module がその
唯一の正本である。

タイムゾーンは CLAUDE.md の「内部 UTC、表示は Asia/Tokyo」に従い UTC に統一する。
ローカル時刻で採番していた呼び出し元のファイル名は 9 時間ぶんずれる。

置き場について:
    利用者が backtest BC と screening BC に跨るため `src/utils/` に置く。
    backtest 配下に置くと screening からの参照が BC 独立性契約に違反する。

設計: docs/superpowers/specs/2026-09-21-execution-model-design.md
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Mapping

import pandas as pd

from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def results_timestamp() -> str:
    """結果ファイル名に使う UTC タイムスタンプ（YYYYmmdd_HHMMSS）を返す。"""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_csv(
    df: pd.DataFrame,
    out_dir: str,
    filename: str,
    encoding: str = "utf-8",
) -> str:
    """out_dir を作成して CSV を書き、保存先パスを返す。

    Args:
        df: 保存する DataFrame
        out_dir: 保存先ディレクトリ（存在しなければ作成する）
        filename: ファイル名（拡張子込み）
        encoding: 文字コード。Excel 互換が要る場合のみ "utf-8-sig" を指定する。
    """
    ensure_dir(out_dir)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False, encoding=encoding)
    return path


def save_result_csvs(
    frames: Mapping[str, pd.DataFrame],
    subdir: str,
    encoding: str = "utf-8",
) -> dict[str, str]:
    """results/<subdir>/<stem>_<UTC タイムスタンプ>.csv に複数の CSV を保存する。

    同じ呼び出しで保存される CSV は同一のタイムスタンプで揃う。

    Args:
        frames: {ファイル名の stem: DataFrame}。stem にタイムスタンプは含めない。
        subdir: results/ からの相対ディレクトリ（例 "backtest/longterm"）
        encoding: 文字コード

    Returns:
        {frames のキー: 保存先パス}
    """
    if not frames:
        return {}

    out_dir = os.path.join(get_results_dir(), *subdir.split("/"))
    ts = results_timestamp()
    paths = {
        key: save_csv(df, out_dir, f"{key}_{ts}.csv", encoding=encoding)
        for key, df in frames.items()
    }
    logger.info("結果を保存: %s", " / ".join(paths.values()))
    return paths
