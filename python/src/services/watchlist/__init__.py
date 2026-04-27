"""
ウォッチリストドメイン（Watchlist Domain）

銘柄ウォッチリストの取得・差分検出・更新に関する公開 API。
DDD への移行を見据えた境界コンテキストの入口。

# 利用例
    from src.services.watchlist import run_watchlist_refresh
"""

from src.services.watchlist.watchlist_manager import (  # noqa: F401
    WatchlistDiff,
    apply_watchlist_update,
    diff_watchlist,
    fetch_index_symbols,
    fetch_nikkei225_symbols,
    fetch_sp500_symbols,
    run_watchlist_refresh,
    verify_symbol_active,
)

__all__ = [
    "WatchlistDiff",
    "fetch_sp500_symbols",
    "fetch_nikkei225_symbols",
    "fetch_index_symbols",
    "verify_symbol_active",
    "diff_watchlist",
    "apply_watchlist_update",
    "run_watchlist_refresh",
]
