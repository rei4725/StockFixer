"""
データ取得・処理に関する設定定数

市場データ取得や予測パイプラインの並列実行パラメータを一元管理する。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # yfinance API 制限を考慮したバッチ取得の並列数
    MAX_DATA_WORKERS: int = Field(default=3)

    # 予測パイプラインのワーカー数（1=同期実行、スレッド間競合を回避）
    MAX_PREDICTION_WORKERS: int = Field(default=1)


data_settings = DataSettings()

MAX_DATA_WORKERS: int = data_settings.MAX_DATA_WORKERS
MAX_PREDICTION_WORKERS: int = data_settings.MAX_PREDICTION_WORKERS
