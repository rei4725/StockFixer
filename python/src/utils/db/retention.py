"""診断ログテーブルの retention（保持期間管理）。

学習・診断のたびに単調増加し DB 肥大化の主因になる診断系テーブルの古い行を削除する。
読み出し側はいずれも各グループの「最新行」しか参照しないため、古い行の削除は安全。

| テーブル | グループキー | 時刻カラム | 読み出し側 |
|---|---|---|---|
| shap_values | market, symbol, model_name | trained_at | MAX(trained_at) のみ参照 |
| feature_selection_log | market, symbol, model_name | trained_at | MAX(trained_at) のみ参照 |
| model_metrics | market, symbol, model_name | trained_at | 最新 trained_at のみ参照（load_model_weights） |
| data_quality_log | market, symbol, check_name | checked_at | 書き込み専用（読み出しなし） |

ただし「保持日数より古い＝全削除」では、長期間更新されていないグループの最新行まで
消えて読み出しが壊れる。そこで **各グループの最新時刻は常に残し**、それ以外で保持日数
より古い行のみを削除する。

**experiment_runs は対象外**: 読み出し側 load_best_run() がメトリクス最良ランを
全履歴から選ぶため（最新行参照ではない）、古い行を消すと挙動が壊れる。実験記録は
監査・分析用に履歴を保持する。

時刻カラムはフォーマットがテーブルごとに異なる:
- trained_at は ``YYYYMMDD_HHMMSS``（compact）
- checked_at は ``datetime.isoformat()``（ISO 8601）

どちらも書式内では辞書順 = 時系列順になるため、同一フォーマットの cutoff 文字列との
比較で正しく新旧判定できる。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 時刻フォーマット種別
FMT_COMPACT = "compact"  # YYYYMMDD_HHMMSS（trained_at）
FMT_ISO = "iso"  # datetime.isoformat()（checked_at）

# (テーブル名, グループキー列, 時刻カラム, 時刻フォーマット)
# — 各グループの最新時刻行は常に保持する
_LOG_TABLES: list[tuple[str, tuple[str, ...], str, str]] = [
    ("shap_values", ("market", "symbol", "model_name"), "trained_at", FMT_COMPACT),
    ("feature_selection_log", ("market", "symbol", "model_name"), "trained_at", FMT_COMPACT),
    ("model_metrics", ("market", "symbol", "model_name"), "trained_at", FMT_COMPACT),
    ("data_quality_log", ("market", "symbol", "check_name"), "checked_at", FMT_ISO),
]


def _cutoff(retention_days: int, now: datetime | None = None, fmt: str = FMT_COMPACT) -> str:
    """時刻カラムと比較する cutoff 文字列を、指定フォーマットで返す。"""
    base = now or datetime.now(timezone.utc)
    cut = base - timedelta(days=retention_days)
    if fmt == FMT_ISO:
        return cut.isoformat()
    return cut.strftime("%Y%m%d_%H%M%S")


def purge_old_training_logs(
    con: Any,
    retention_days: int,
    now: datetime | None = None,
) -> dict[str, int]:
    """各診断ログテーブルの古い行を削除し、テーブルごとの削除件数を返す。

    Args:
        con: DuckDB 接続（書き込み可能なもの）。
        retention_days: 保持日数。これより古い行を削除（各グループの最新を除く）。
        now: 基準時刻（テスト用。省略時は現在 UTC）。

    Returns:
        {テーブル名: 削除行数}
    """
    if retention_days < 0:
        raise ValueError("retention_days は 0 以上が必要です")

    deleted: dict[str, int] = {}

    for table, keys, time_col, fmt in _LOG_TABLES:
        cutoff = _cutoff(retention_days, now, fmt)
        key_join = " AND ".join(f't2.{k} = "{table}".{k}' for k in keys)
        # 保持日数より古く、かつ自グループの最新時刻ではない行のみ削除
        where = (
            f"{time_col} < %s "
            f"AND {time_col} < "
            f'(SELECT MAX(t2.{time_col}) FROM "{table}" t2 WHERE {key_join})'
        )
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {where}', [cutoff]).fetchone()[0]
            if n:
                con.execute(f'DELETE FROM "{table}" WHERE {where}', [cutoff])
            deleted[table] = int(n)
            logger.info("retention: %s から %d 行削除 (cutoff=%s)", table, n, cutoff)
        except Exception as e:
            logger.warning("retention 失敗 %s: %s", table, e, exc_info=True)
            deleted[table] = 0

    return deleted
