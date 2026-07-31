"""予測配信サービスのリクエスト/レスポンス型。

本体（src/）の PredictionResult とは意図的に分離している。API コントラクトは
サービス間の契約であり、本体の内部型が変わっても壊れないようにするため。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """推論リクエスト。

    特徴量・現在価格・モデル重みはすべて呼び出し側（本体）が用意する。
    サービスは DB にも yfinance にもアクセスしないため、推論に必要な情報は
    すべてこのリクエストに含まれている必要がある。
    """

    market: str
    symbol: str
    current_price: float
    features: dict[str, float] = Field(min_length=1)
    model_types: list[str] = Field(min_length=1)
    model_weights: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_weights_length(self) -> PredictRequest:
        if len(self.model_weights) != len(self.model_types):
            raise ValueError(
                f"model_weights の要素数({len(self.model_weights)})が "
                f"model_types の要素数({len(self.model_types)})と一致しません"
            )
        return self


class PredictResponse(BaseModel):
    """推論レスポンス。

    全モデルの推論が失敗した場合も HTTP 200 を返し、model_count=0 で表現する
    （本体側はこれを「予測なし」として扱う）。
    """

    market: str
    symbol: str
    current_price: float
    avg_pred_price: float
    diff_ratio: float
    model_count: int
    used_models: list[str]


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス。"""

    status: str
    loaded_models: list[str]
