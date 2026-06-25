"""
Wikipedia ページビュー特徴量モジュール (#370: 無料オルタナデータ adapter)

銘柄に対応する Wikipedia 記事の日次ページビューを取得し、OHLCV DataFrame に
「一般の注目度」を表す特徴量として付加する。注目度の急増は出来高・ボラティリティの
先行シグナルになり得る（学術研究で有意性が確認されている）。

公開 API:
    fetch_pageview_features(symbol, market, start_date, end_date) -> Optional[pd.DataFrame]
        Wikimedia API から {date: views} を取得し DataFrame(列: pageviews) に変換する。

    add_pageview_features(df, pageview_df) -> pd.DataFrame
        追加列:
          - pageviews          : その日のページビュー数（欠損日は 0）
          - pageviews_ma{w}    : window_days 日移動平均
          - pageviews_zscore   : 直近ベースラインからの偏差（注目度スパイク検出）
          - pageviews_momentum : 前日比変化率

センチメント特徴量（sentiment_features.py）と同様、外部データが取得できない場合でも
列は常に付与し中立値（0）で埋めることで、統合モデルのスキーマを安定させる。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 移動平均のウィンドウ幅（日）
_DEFAULT_WINDOW = 7
# zscore のベースライン算出ウィンドウ幅（日）
_ZSCORE_WINDOW = 30


def fetch_pageview_features(
    symbol: str,
    market: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    """Wikimedia API から日次ページビューを取得し DataFrame に変換する。

    Args:
        symbol: 銘柄コード（例: "AAPL", "7203"）
        market: 市場識別子 ("us" / "jp")
        start_date: 取得開始日 (YYYY-MM-DD)
        end_date: 取得終了日 (YYYY-MM-DD)

    Returns:
        DatetimeIndex を持つ DataFrame (列: pageviews)。
        記事未登録・データなし・取得失敗の場合は None。
    """
    from src.market_data.adapters.wikipedia_pageviews import WikipediaPageviewsClient

    try:
        client = WikipediaPageviewsClient()
        records = client.fetch_daily_views(market, symbol, start_date, end_date)
    except Exception as e:
        logger.error("Wikipedia ページビュー取得エラー: %s", e, exc_info=True)
        return None

    if not records:
        return None

    rows = [
        {"date": pd.Timestamp(date_str), "pageviews": views} for date_str, views in records.items()
    ]
    return pd.DataFrame(rows).set_index("date").sort_index()


def add_pageview_features(
    df: pd.DataFrame,
    pageview_df: Optional[pd.DataFrame] = None,
    window_days: int = _DEFAULT_WINDOW,
) -> pd.DataFrame:
    """OHLCV + テクニカル指標 DataFrame に Wikipedia ページビュー特徴量を追加する。

    追加列:
      - pageviews          : その日のページビュー数（欠損日は 0）
      - pageviews_ma{w}    : window_days 日移動平均
      - pageviews_zscore   : 直近 30 日ベースラインからの標準化偏差（注目度スパイク）
      - pageviews_momentum : 前日比変化率（前日 0 のときは 0）

    Args:
        df: DatetimeIndex を持つ OHLCV + テクニカル指標 DataFrame
        pageview_df: 事前取得済みの (pageviews) DataFrame。
                     None の場合は中立プレースホルダー（0）を使用する。
        window_days: 移動平均のウィンドウ幅（デフォルト: 7 日）

    Returns:
        ページビュー特徴量列を追加した DataFrame（入力コピー）
    """
    df = df.copy()

    if pageview_df is not None and not pageview_df.empty:
        pv = pageview_df.reindex(df.index)
        df["pageviews"] = pv["pageviews"].fillna(0.0).astype(float)
        logger.debug(
            "Wikipedia ページビュー特徴量を付与 (外部データ, %d件)",
            int(pv["pageviews"].notna().sum()),
        )
    else:
        df["pageviews"] = 0.0
        logger.debug("Wikipedia ページビュー特徴量をプレースホルダー (0) で初期化")

    df[f"pageviews_ma{window_days}"] = (
        df["pageviews"].rolling(window=window_days, min_periods=1).mean()
    )

    rolling = df["pageviews"].rolling(window=_ZSCORE_WINDOW, min_periods=1)
    baseline = rolling.mean()
    std = rolling.std()
    zscore = (df["pageviews"] - baseline) / std
    df["pageviews_zscore"] = zscore.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    momentum = df["pageviews"].pct_change()
    df["pageviews_momentum"] = momentum.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df
