"""ニュースセンチメントの取得元のキーと、センチメントの特徴量の列の決まり（#1417）。

米国株は NewsAPI（NEWSAPI_KEY）、日本株は EDINET（EDINET_API_KEY）から記事・開示を取る。
キーが無い市場ではセンチメントの列に値が一度も入らないため、特徴量から外す。
"""

import os
from typing import Iterable, Mapping, Optional

SENTIMENT_FEATURE_PREFIXES = ("sentiment_", "news_count")

SENTIMENT_KEY_BY_MARKET = {"us": "NEWSAPI_KEY", "jp": "EDINET_API_KEY"}


def sentiment_source_configured(market: str, env: Optional[Mapping[str, str]] = None) -> bool:
    """market のセンチメントの取得元のキーが空でなく設定されていれば True。"""
    key_name = SENTIMENT_KEY_BY_MARKET.get(str(market))
    if key_name is None:
        return False
    values = os.environ if env is None else env
    return bool(values.get(key_name, "").strip())


def any_sentiment_source_configured(
    markets: Iterable[str], env: Optional[Mapping[str, str]] = None
) -> bool:
    """markets のどれか1つでも取得元のキーがあれば True。"""
    return any(sentiment_source_configured(m, env) for m in markets)


def is_sentiment_feature(column: str) -> bool:
    return str(column).startswith(SENTIMENT_FEATURE_PREFIXES)
