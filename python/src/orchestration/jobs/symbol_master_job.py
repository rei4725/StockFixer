"""銘柄マスタ（symbol_master）の週の更新（#1080）。

weekly.py からの責務分離（ファイル行数ガード対策）。週の DB メンテナンスの最後に呼ぶ。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_symbol_master_refresh() -> None:
    """watchlist.json の jp 銘柄の社名を JPX 一覧から埋め直す。

    取得や保存が落ちても WARNING を残して戻り、呼んだ週のジョブのほかの処理は止めない。
    """
    try:
        from src.market_data.symbol_master import refresh_jp_symbol_master

        refresh_jp_symbol_master()
    except Exception as e:
        logger.warning("銘柄マスタ更新失敗（週のジョブは続ける）: %s", e, exc_info=True)
