"""
ニュースセンチメント特徴量モジュール

外部データソース（NewsAPI 等）からニュースセンチメントスコアを取得し、
OHLCV DataFrame に特徴量として付加する。

データソース優先順位:
  1. 呼び出し元が渡した sentiment_df（外部取得済みデータ）
  2. NEWSAPI_KEY 環境変数が設定されていれば NewsAPI v2 から取得
  3. どちらも利用できない場合は中立スコア（0.0）でプレースホルダーを返す

スコアリング方式（fetch_news_sentiment_with_llm を使用した場合）:
  - OLLAMA_URL 設定済み + Ollama 応答あり → LLM スコアリング
  - それ以外 → キーワードマッチにフォールバック

将来拡張:
  - HuggingFace FinBERT によるオフライン推論
  - FRED 経済指標（UMCSENT 等）との融合
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# センチメント取得に失敗した場合の中立スコア
_NEUTRAL_SCORE = 0.0


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------


def _fetch_news_articles(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Optional[dict[str, list[str]]]:
    """NewsAPI v2 から日付別のタイトルリストを取得する。

    Returns:
        {date_str: [title, ...]} 形式の辞書。取得不可の場合は None。
    """
    api_key = os.environ.get("NEWSAPI_KEY")
    if not api_key:
        logger.debug("NEWSAPI_KEY が未設定のためセンチメント API 取得をスキップ")
        return None

    try:
        import requests  # noqa: PLC0415

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": start_date,
            "to": end_date,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)  # type: ignore[arg-type]
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        if not articles:
            return None

        records: dict[str, list[str]] = {}
        for article in articles:
            published = article.get("publishedAt", "")[:10]
            if not published:
                continue
            records.setdefault(published, []).append(article.get("title", "") or "")

        return records if records else None

    except Exception as e:
        logger.error("NewsAPI 取得エラー: %s", e, exc_info=True)
        return None


def _records_to_sentiment_df(
    records: dict[str, list[str]],
    score_fn: Callable[[list[str]], float],
) -> Optional[pd.DataFrame]:
    """日付別タイトル辞書をスコアリングして sentiment DataFrame に変換する。"""
    rows = []
    for date_str, titles in records.items():
        score = score_fn(titles)
        rows.append(
            {
                "date": pd.Timestamp(date_str),
                "sentiment_score": score,
                "news_count": len(titles),
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows).set_index("date").sort_index()


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def fetch_news_sentiment(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """NewsAPI v2 + キーワードマッチでセンチメントスコアを取得する。

    後方互換用。新規呼び出しは fetch_news_sentiment_with_llm を推奨。

    Returns:
        DatetimeIndex を持つ DataFrame (列: sentiment_score, news_count)。
        取得不可の場合は None。
    """
    records = _fetch_news_articles(symbol, start_date, end_date)
    if records is None:
        return None
    return _records_to_sentiment_df(records, _score_titles)


def fetch_news_sentiment_with_llm(
    symbol: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """NewsAPI v2 からニュースを取得し、Ollama LLM でセンチメントスコアを算出する。

    OLLAMA_URL が未設定または Ollama に接続できない場合は、キーワードマッチに
    自動フォールバックするため、既存環境でも安全に呼び出せる。

    Returns:
        DatetimeIndex を持つ DataFrame (列: sentiment_score, news_count)。
        NewsAPI キーなし / 記事なし の場合は None。
    """
    records = _fetch_news_articles(symbol, start_date, end_date)
    if records is None:
        return None

    from src.market_data.adapters.llm_sentiment import OllamaClient  # noqa: PLC0415

    client = OllamaClient()

    if client.is_available:
        logger.debug("Ollama LLM でセンチメントスコアリング")

        def _llm_score(titles: list[str]) -> float:
            score = client.score(titles)
            if score is None:
                # LLM が None を返した場合（タイムアウト等）はキーワードマッチで補完
                return _score_titles(titles)
            return score

        score_fn: Callable[[list[str]], float] = _llm_score
    else:
        logger.debug("Ollama 未接続のためキーワードマッチにフォールバック")
        score_fn = _score_titles

    return _records_to_sentiment_df(records, score_fn)


def _score_titles(titles: list[str]) -> float:
    """タイトルリストから簡易センチメントスコア（-1.0〜1.0）を算出する。

    FinBERT や VADER が利用可能な環境では差し替え可能。
    ここでは正/負キーワードの出現比率から算出する簡易実装。
    """
    _POSITIVE = {"surge", "jump", "beat", "record", "rally", "gain", "rise", "bullish", "strong"}
    _NEGATIVE = {"fall", "drop", "miss", "decline", "crash", "plunge", "bearish", "weak", "cut"}

    pos = neg = 0
    for title in titles:
        lower = title.lower()
        pos += sum(1 for w in _POSITIVE if w in lower)
        neg += sum(1 for w in _NEGATIVE if w in lower)

    total = pos + neg
    if total == 0:
        return _NEUTRAL_SCORE
    return round((pos - neg) / total, 4)


def add_sentiment_features(
    df: pd.DataFrame,
    sentiment_df: Optional[pd.DataFrame] = None,
    window_days: int = 5,
) -> pd.DataFrame:
    """OHLCV + テクニカル指標 DataFrame にセンチメント特徴量を追加する。

    追加列:
      - sentiment_score    : その日のセンチメントスコア (-1.0〜1.0, 欠損日は 0.0)
      - sentiment_ma{w}    : window_days 日移動平均スコア
      - sentiment_momentum : 前日比スコア変化量
      - news_count         : その日のニュース件数 (欠損日は 0)

    Args:
        df: DatetimeIndex を持つ OHLCV + テクニカル指標 DataFrame
        sentiment_df: 事前取得済みの (sentiment_score, news_count) DataFrame。
                      None の場合は中立プレースホルダーを使用する。
        window_days: 移動平均のウィンドウ幅（デフォルト: 5 日）

    Returns:
        センチメント特徴量列を追加した DataFrame（入力コピー）
    """
    df = df.copy()

    if sentiment_df is not None and not sentiment_df.empty:
        sent = sentiment_df.reindex(df.index)
        df["sentiment_score"] = sent["sentiment_score"].fillna(_NEUTRAL_SCORE)
        df["news_count"] = sent["news_count"].fillna(0).astype(int)
        logger.debug(
            "センチメント特徴量を付与 (外部データ, %d件)",
            int(sent["sentiment_score"].notna().sum()),
        )
    else:
        df["sentiment_score"] = _NEUTRAL_SCORE
        df["news_count"] = 0
        logger.debug("センチメント特徴量をプレースホルダー (中立値) で初期化")

    df[f"sentiment_ma{window_days}"] = (
        df["sentiment_score"].rolling(window=window_days, min_periods=1).mean()
    )
    df["sentiment_momentum"] = df["sentiment_score"].diff().fillna(0.0)

    return df
