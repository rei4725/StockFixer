"""
統一ロガーファクトリー

全レイヤーで使用する標準ロガーを提供する。
- Logs/stockfixer.log       : 全ログ（INFO以上、10MB × 5世代）
- Logs/stockfixer_error.log : エラーログ（ERROR以上、5MB × 3世代）
- stderr                    : INFO以上をコンソール出力

使い方:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("処理開始")
    logger.error("エラー発生", exc_info=True)

ログレベルは環境変数 LOG_LEVEL（デフォルト: INFO）で制御可能。
ログ形式は環境変数 LOG_FORMAT（json / 未設定）で制御可能。

【機密値マスキング】
以下の条件を満たす環境変数の値はログ出力前に自動的に ***REDACTED*** へ置換される。
  - 変数名が DISCORD_, TOKEN, SECRET, PASSWORD, WEBHOOK, API_KEY, AUTH,
    CREDENTIAL のいずれかを含む
  - 値の長さが 8 文字以上（短すぎる値による誤検知を防止）
例外トレースバック内の機密値も対象。
"""

import datetime
import io
import json
import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_LOG_FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(name)s] [%(run_id)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# リポジトリルートディレクトリ（src/utils/logger.py から4階層上）
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_LOG_DIR = os.path.join(_REPO_ROOT, "Logs")

_root_configured = False

# ── 機密値マスキング ─────────────────────────────────────────────────────────
_MASK_PLACEHOLDER = "***REDACTED***"
_MIN_MASK_LENGTH = 8

# 機密性が高い環境変数名のキーワード（大文字で比較）
_SENSITIVE_KEY_WORDS = (
    "DISCORD_",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "WEBHOOK",
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
)


def _collect_sensitive_values() -> list:
    """
    環境変数から機密値を収集して返す。

    - 変数名が _SENSITIVE_KEY_WORDS のいずれかを含むもののみ対象
    - 長さが _MIN_MASK_LENGTH 未満の値は除外（誤検知防止）
    - 長い値を先に置換するため降順ソート（部分文字列の誤置換防止）
    """
    values = []
    for key, val in os.environ.items():
        if len(val) < _MIN_MASK_LENGTH:
            continue
        upper_key = key.upper()
        if any(kw in upper_key for kw in _SENSITIVE_KEY_WORDS):
            values.append(val)
    return sorted(set(values), key=len, reverse=True)


class _MaskingFormatter(logging.Formatter):
    """
    機密値をログ出力前にマスクするフォーマッター。
    super().format() で生成した最終文字列（例外トレースバック含む）に対して
    一括置換するため、あらゆる出力経路をカバーできる。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sensitive = _collect_sensitive_values()
        if sensitive:
            pattern = "|".join(re.escape(v) for v in sensitive)
            self._pattern: Optional[re.Pattern] = re.compile(pattern)
        else:
            self._pattern = None

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "run_id"):
            from src.utils.run_context import get_run_id

            record.run_id = get_run_id() or "-"
        formatted = super().format(record)
        if self._pattern:
            formatted = self._pattern.sub(_MASK_PLACEHOLDER, formatted)
        return formatted


class JsonFormatter(logging.Formatter):
    """
    LOG_FORMAT=json のとき使用する JSON フォーマッター。
    ELK / Loki 等のログ集約基盤への取り込みを前提とした1行 JSON を出力する。
    機密値マスキングも _MaskingFormatter と同等に適用される。
    """

    def __init__(self):
        super().__init__()
        sensitive = _collect_sensitive_values()
        if sensitive:
            pattern = "|".join(re.escape(v) for v in sensitive)
            self._pattern: Optional[re.Pattern] = re.compile(pattern)
        else:
            self._pattern = None

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "run_id"):
            from src.utils.run_context import get_run_id

            record.run_id = get_run_id() or "-"

        ts = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        ts = f"{ts}.{int(record.msecs):03d}Z"

        exc_info_str: Optional[str] = None
        if record.exc_info:
            exc_info_str = self.formatException(record.exc_info)

        log_entry = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "exc_info": exc_info_str,
        }

        result = json.dumps(log_entry, ensure_ascii=False)
        if self._pattern:
            result = self._pattern.sub(_MASK_PLACEHOLDER, result)
        return result


# ─────────────────────────────────────────────────────────────────────────────


def _configure_root() -> None:
    """ルートロガーを一度だけ設定する（以降の呼び出しはno-op）"""
    global _root_configured
    if _root_configured:
        return

    os.makedirs(_LOG_DIR, exist_ok=True)

    log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_format = os.environ.get("LOG_FORMAT", "").lower()
    if log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = _MaskingFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

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
