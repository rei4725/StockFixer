"""
ジョブ実行に関する設定定数

スケジューラが参照する環境変数を BaseSettings で一元管理する。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # 週次ルール評価ジョブのパラメータ
    RULE_EVAL_MARKET: str = Field(default="jp")
    RULE_EVAL_START: str = Field(default="2022-01-01")
    RULE_EVAL_END: str = Field(default="2025-01-01")


execution_settings = ExecutionSettings()

RULE_EVAL_MARKET: str = execution_settings.RULE_EVAL_MARKET
RULE_EVAL_START: str = execution_settings.RULE_EVAL_START
RULE_EVAL_END: str = execution_settings.RULE_EVAL_END
