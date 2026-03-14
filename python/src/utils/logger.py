"""
統一ロガーファクトリー

全レイヤーで使用する標準ロガーを提供する。
- python/logs/stockfixer.log       : 全ログ（INFO以上、10MB × 5世代）
- python/logs/stockfixer_error.log : エラーログ（ERROR以上、5MB × 3世代）
- stderr                           : INFO以上をコンソール出力

使い方:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("処理開始")
    logger.error("エラー発生", exc_info=True)

ログレベルは環境変数 LOG_LEVEL（デフォルト: INFO）で制御可能。
"""

import io
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# python/ ルートディレクトリ（src/utils/logger.py から3階層上）
_PYTHON_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOG_DIR = os.path.join(_PYTHON_ROOT, "logs")

_root_configured = False


def _configure_root() -> None:
    """ルートロガーを一度だけ設定する（以降の呼び出しはno-op）"""
    global _root_configured
    if _root_configured:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(log_level)

    # ハンドラ重複防止（basicConfig等で既に設定済みの場合はスキップ）
    if root.handlers:
        _root_configured = True
        return

    # ① 全ログファイル（INFO以上、10MB × 5世代）
    all_handler = RotatingFileHandler(
        os.path.join(_LOG_DIR, "stockfixer.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    all_handler.setLevel(log_level)
    all_handler.setFormatter(formatter)
    root.addHandler(all_handler)

    # ② エラーログファイル（ERROR以上、5MB × 3世代）
    error_handler = RotatingFileHandler(
        os.path.join(_LOG_DIR, "stockfixer_error.log"),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root.addHandler(error_handler)

    # ③ コンソール出力（stderr、INFO以上・UTF-8強制）
    _utf8_stderr = (
        io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        if hasattr(sys.stderr, "buffer")
        else sys.stderr
    )
    console_handler = logging.StreamHandler(_utf8_stderr)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # 外部ライブラリのノイズを抑制
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    統一設定済みのロガーを返す。

    Args:
        name: ロガー名。通常は __name__ を渡す。

    Returns:
        logging.Logger: 設定済みのロガーインスタンス

    使い方:
        logger = get_logger(__name__)
    """
    _configure_root()
    return logging.getLogger(name)
