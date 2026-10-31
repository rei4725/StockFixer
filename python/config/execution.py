"""
ジョブ実行に関する設定定数

スケジューラが参照する環境変数を BaseSettings で一元管理する。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.env_source import dotenv_path


class ExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=dotenv_path(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # 週次ルール評価ジョブのパラメータ
    RULE_EVAL_MARKET: str = Field(default="jp")
    # 期間は既定で実行日までの直近 RULE_EVAL_YEARS 年。START/END を入れたときだけ固定する
    RULE_EVAL_YEARS: int = Field(default=3, ge=1)
    RULE_EVAL_START: str | None = Field(default=None)
    RULE_EVAL_END: str | None = Field(default=None)


execution_settings = ExecutionSettings()

RULE_EVAL_MARKET: str = execution_settings.RULE_EVAL_MARKET
RULE_EVAL_YEARS: int = execution_settings.RULE_EVAL_YEARS
RULE_EVAL_START: str | None = execution_settings.RULE_EVAL_START
RULE_EVAL_END: str | None = execution_settings.RULE_EVAL_END
