r"""DuckDB 物理コンパクション CLI（肥大化した DB の容量回収）。

VACUUM ではファイルが縮まないため、生存行だけを新ファイルへ再構築コピーして
容量を回収する。**実行前にコンテナ（DB を書き込み中のプロセス）を停止すること。**

使用例:
    # まずドライラン（新ファイルを作って行数・サイズを表示するだけ。入れ替えはしない）
    py run_db_compact.py

    # 検証OKなら入れ替え（元ファイルは .bak-<timestamp> として退避）
    py run_db_compact.py --swap

    # 保持日数を指定（診断ログをより短く絞る）
    py run_db_compact.py --retention-days 14 --swap
"""

import argparse
import os
import sys

from config.settings import DB_LOG_RETENTION_DAYS
from src.utils.data_path_utils import get_db_path
from src.utils.db.compact import compact_database, swap_compacted
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def parse_args():
    p = argparse.ArgumentParser(description="DuckDB を再構築して容量を回収する")
    p.add_argument(
        "--retention-days",
        type=int,
        default=DB_LOG_RETENTION_DAYS,
        help=f"診断ログの保持日数（既定 {DB_LOG_RETENTION_DAYS}）",
    )
    p.add_argument("--swap", action="store_true", help="検証後に元ファイルと入れ替える")
    return p.parse_args()


def main():
    args = parse_args()
    db_path = get_db_path()
    if not os.path.exists(db_path):
        logger.error("DB が見つかりません: %s", db_path)
        sys.exit(1)

    new_path = db_path + ".compact"
    if os.path.exists(new_path):
        os.remove(new_path)

    size_before = _mb(db_path)
    logger.info(
        "コンパクション開始: %s (%.1f MB) retention=%d日", db_path, size_before, args.retention_days
    )

    try:
        counts = compact_database(db_path, new_path, args.retention_days)
    except Exception as e:
        logger.error(
            "コンパクション失敗（コンテナ停止済みか確認してください）: %s", e, exc_info=True
        )
        if os.path.exists(new_path):
            os.remove(new_path)
        sys.exit(1)

    size_after = _mb(new_path)
    print("\n=== テーブル別 行数（元 -> コピー後） ===")
    for table, (orig_n, new_n) in sorted(counts.items(), key=lambda kv: -kv[1][0]):
        mark = "  *削減" if new_n < orig_n else ""
        print(f"  {table:<28} {orig_n:>12,} -> {new_n:>12,}{mark}")
    print(
        f"\nファイルサイズ: {size_before:,.1f} MB -> {size_after:,.1f} MB "
        f"（{(1 - size_after / size_before) * 100:.1f}% 削減）"
        if size_before
        else ""
    )

    if not args.swap:
        print(f"\n[dry-run] 新ファイルを作成しました: {new_path}")
        print("問題なければ --swap で入れ替えてください（元は .bak-<時刻> に退避されます）。")
        return

    # 入れ替え（元ファイルは退避）
    bak_path = swap_compacted(db_path, new_path, keep_backup=True)
    logger.info("入れ替え完了: 旧ファイルを %s に退避", bak_path)
    print(f"\n✅ 入れ替え完了。元ファイルは {bak_path} に退避（検証後に削除可）。")
    print("コンテナを起動して health を確認してください。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical("コンパクション異常終了: %s", e, exc_info=True)
        sys.exit(1)
