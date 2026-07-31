# 予測配信マイクロサービス Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 予測配信（推論）を独立したHTTPサービスとして切り出し、本体から既定OFF・フォールバック付きで呼び出せるようにする（学習目的のフェーズ1）。

**Architecture:** 本体（モノリス）がDBとyfinanceからデータを取得し、純粋な計算のみを行う推論サービスへHTTPで渡す。推論サービスはDB接続もyfinance接続も持たず、`python/models/unified/` の joblib ファイルのみを読む。環境変数 `PREDICTION_SERVICE_URL` 未設定時は従来のインプロセス推論が動き、設定時もサービス障害時は自動でインプロセスへフォールバックする。

**Tech Stack:** FastAPI（推論サービス）, Pydantic（リクエスト/レスポンス型）, requests（本体側クライアント、既存依存）, pytest + FastAPI TestClient

**設計書:** `docs/superpowers/specs/2026-07-31-prediction-microservice-design.md`

## Global Constraints

- 推論サービスは Postgres への接続情報を一切持たない（`src.utils.db` を import しない）
- 推論サービスは yfinance を呼ばない（外部I/Oゼロ、モデルファイル読み込みのみ）
- `PREDICTION_SERVICE_URL` 未設定が既定動作であり、そのとき既存の挙動と完全に同一であること
- サービス呼び出しの失敗（接続不可 / タイムアウト / 5xx）は例外を伝播させず、警告ログを出してインプロセス推論へフォールバックする
- `services/` は `python/src/` の外に置く（import-linter のレイヤー契約の対象外とするため）
- 本体の `src/` 配下から `services/` を import してはならない（逆方向も同様）
- サービスの依存はすべて `requirements-service.txt` に記述する（`requirements.txt` は変更しない）。ML ライブラリ（scikit-learn / xgboost / lightgbm / pandas）は joblib モデルの互換性のため **`requirements.txt` と同一のバージョンピン**を使う。Dockerfile にバージョンを直書きしないこと
- 全モデルの推論が失敗した場合、HTTPステータスは 200 のまま `model_count: 0` で表現する
- Conventional Commits 規約に従う（`feat:` / `test:` / `docs:` 等）
- 本番設定（`python/.env`, `docker-compose.yml`）は今回変更しない

---

### Task 1: 推論サービスの型定義とスケルトン

**Files:**
- Create: `python/services/__init__.py`
- Create: `python/services/prediction_service/__init__.py`
- Create: `python/services/prediction_service/types.py`
- Create: `python/requirements-service.txt`
- Test: `python/tests/unit/services/__init__.py`（空ファイル）
- Test: `python/tests/unit/services/test_prediction_types.py`

**Interfaces:**
- Consumes: なし（最初のタスク）
- Produces:
  - `PredictRequest` (Pydantic BaseModel): `market: str`, `symbol: str`, `current_price: float`, `features: dict[str, float]`, `model_types: list[str]`, `model_weights: list[float]`
  - `PredictResponse` (Pydantic BaseModel): `market: str`, `symbol: str`, `current_price: float`, `avg_pred_price: float`, `diff_ratio: float`, `model_count: int`, `used_models: list[str]`
  - `HealthResponse` (Pydantic BaseModel): `status: str`, `loaded_models: list[str]`

- [ ] **Step 1: 依存ファイルを作成**

`python/requirements-service.txt` を作成:

```
# 予測配信マイクロサービス専用の依存定義。
# fastapi/uvicorn は本体（requirements.txt）には含めない — サービスは別イメージで
# デプロイするため。
fastapi==0.120.1
uvicorn==0.38.0

# --- モデル読み込み・推論に必要な依存 ---
# joblib モデルは本体が保存しサービスが読むため、バージョンがずれると
# アンピクル失敗や推論結果の不一致を招く。requirements.txt と必ず同じピンを
# 使うこと（本体側を更新する際はこちらも同時に更新する）。
scikit-learn==1.9.0
lightgbm==4.7.0
xgboost==3.3.0
pandas>=3.0.3,<4
joblib>=1.4
```

- [ ] **Step 2: パッケージの __init__.py を作成**

`python/services/__init__.py`:

```python
"""独立デプロイされるマイクロサービス群。

本体（src/）とは別プロセス・別イメージで動くため、src/ の import-linter
レイヤー契約の外側に置く。src/ からこのパッケージを import してはならない。
"""
```

`python/services/prediction_service/__init__.py`:

```python
"""予測配信マイクロサービス（学習用フェーズ1）。

特徴量と現在価格を受け取り、モデル推論とアンサンブルのみを行う純粋な
計算サービス。DB接続・yfinance接続を持たない。
"""
```

- [ ] **Step 3: 失敗するテストを書く**

`python/tests/unit/services/__init__.py` は空ファイルとして作成する。

`python/tests/unit/services/test_prediction_types.py`:

```python
"""予測配信サービスのリクエスト/レスポンス型のテスト。"""

import pytest
from pydantic import ValidationError

from services.prediction_service.types import HealthResponse, PredictRequest, PredictResponse


def _valid_request_kwargs() -> dict:
    return {
        "market": "jp",
        "symbol": "7203",
        "current_price": 2500.0,
        "features": {"Close_lag1": 2480.0, "rsi_lag1": 55.2},
        "model_types": ["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
        "model_weights": [0.6, 0.4],
    }


def test_valid_request_accepted():
    req = PredictRequest(**_valid_request_kwargs())
    assert req.market == "jp"
    assert req.symbol == "7203"
    assert req.current_price == 2500.0
    assert req.features["Close_lag1"] == 2480.0
    assert req.model_types == ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
    assert req.model_weights == [0.6, 0.4]


def test_weights_length_mismatch_rejected():
    """model_weights の要素数が model_types と一致しない場合は弾く。"""
    kwargs = _valid_request_kwargs()
    kwargs["model_weights"] = [1.0]
    with pytest.raises(ValidationError):
        PredictRequest(**kwargs)


def test_empty_model_types_rejected():
    kwargs = _valid_request_kwargs()
    kwargs["model_types"] = []
    kwargs["model_weights"] = []
    with pytest.raises(ValidationError):
        PredictRequest(**kwargs)


def test_response_roundtrip():
    resp = PredictResponse(
        market="jp",
        symbol="7203",
        current_price=2500.0,
        avg_pred_price=2537.5,
        diff_ratio=0.015,
        model_count=2,
        used_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
    )
    payload = resp.model_dump()
    assert payload["avg_pred_price"] == 2537.5
    assert payload["model_count"] == 2


def test_health_response():
    health = HealthResponse(status="ok", loaded_models=["UnifiedStockXGBoost"])
    assert health.status == "ok"
    assert health.loaded_models == ["UnifiedStockXGBoost"]
```

- [ ] **Step 4: テストを実行して失敗を確認**

Run: `cd python && py -m pytest tests/unit/services/test_prediction_types.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'services.prediction_service.types'`）

- [ ] **Step 5: 型定義を実装**

`python/services/prediction_service/types.py`:

```python
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
    def _validate_weights_length(self) -> "PredictRequest":
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
```

- [ ] **Step 6: テストを実行して成功を確認**

Run: `cd python && py -m pytest tests/unit/services/test_prediction_types.py -v`
Expected: PASS（5 tests）

- [ ] **Step 7: Lint を通す**

Run: `cd python && py -m black services/ tests/unit/services/ && py -m isort services/ tests/unit/services/ && py -m flake8 services/ tests/unit/services/`
Expected: エラーなし

- [ ] **Step 8: コミット**

```bash
git add python/services/ python/tests/unit/services/ python/requirements-service.txt
git commit -m "feat: 予測配信サービスの型定義とパッケージ構造を追加"
```

---

### Task 2: 推論ロジック（inference.py）

**Files:**
- Create: `python/services/prediction_service/inference.py`
- Test: `python/tests/unit/services/test_inference.py`

**Interfaces:**
- Consumes: Task 1 の `PredictRequest`, `PredictResponse`
- Produces:
  - `load_model(model_name: str) -> Any | None` — モデルをロードしてキャッシュする（見つからなければ None）
  - `run_inference(request: PredictRequest) -> PredictResponse` — 推論本体
  - `clear_model_cache() -> None` — テスト用にキャッシュを空にする
  - `_MODEL_CACHE: dict[str, Any]` — モジュールレベルのキャッシュ（テストから monkeypatch する）

**背景（本体の既存実装）:** `src/prediction/predict_unified.py` の `predict_with_unified_model()` が行っている処理のうち、以下を移植する。

1. モデルの `feature_names_in_` に合わせて不足特徴量を 0 で埋め、順序を揃える
2. 各モデルで `predict()` を呼び、返り値（Series / list / スカラー）から float を取り出す
3. 変化率から絶対価格を計算: `pred_price = current_price * (1 + pred_return)`
4. 重み付き平均: `avg_pred_price = sum(p * w for p, w in zip(pred_prices, weights))`
5. `diff_ratio = (avg_pred_price - current_price) / current_price`

**重要な差分:** 本体では全モデル失敗時に `None` を返すが、サービスでは `model_count=0` の `PredictResponse` を返す。また、成功したモデルだけで重みを再正規化する（本体の既存実装は重み配列をそのまま使うため、一部モデルが失敗すると重みの合計が1未満になるバグがあるが、サービス側では正しく再正規化する）。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/services/test_inference.py`:

```python
"""推論ロジック（inference.py）のテスト。

実モデルファイルは使わず、feature_names_in_ を持つ MagicMock を注入する。
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from services.prediction_service.inference import clear_model_cache, run_inference
from services.prediction_service.types import PredictRequest


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def _make_model(pred_return: float, expected_features: list[str] | None = None):
    """指定した変化率を返すモックモデルを作る。"""
    model = MagicMock()
    inner = MagicMock()
    if expected_features is not None:
        inner.feature_names_in_ = expected_features
    else:
        del inner.feature_names_in_
    model.model = inner
    model.predict.return_value = pd.Series([pred_return])
    return model


def _request(**overrides) -> PredictRequest:
    kwargs = {
        "market": "jp",
        "symbol": "7203",
        "current_price": 1000.0,
        "features": {"a": 1.0, "b": 2.0},
        "model_types": ["ModelA"],
        "model_weights": [1.0],
    }
    kwargs.update(overrides)
    return PredictRequest(**kwargs)


def test_single_model_prediction(monkeypatch):
    """1モデル・変化率0.02 → 予測価格1020, diff_ratio 0.02。"""
    model = _make_model(0.02, expected_features=["a", "b"])
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model", lambda name: model
    )

    resp = run_inference(_request())

    assert resp.model_count == 1
    assert resp.used_models == ["ModelA"]
    assert resp.avg_pred_price == pytest.approx(1020.0)
    assert resp.diff_ratio == pytest.approx(0.02)


def test_weighted_ensemble(monkeypatch):
    """2モデル（+0.02 / +0.04）を重み 0.5/0.5 → 平均 +0.03 相当。"""
    models = {
        "ModelA": _make_model(0.02, expected_features=["a", "b"]),
        "ModelB": _make_model(0.04, expected_features=["a", "b"]),
    }
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model", lambda name: models[name]
    )

    resp = run_inference(
        _request(model_types=["ModelA", "ModelB"], model_weights=[0.5, 0.5])
    )

    assert resp.model_count == 2
    assert resp.avg_pred_price == pytest.approx(1030.0)
    assert resp.diff_ratio == pytest.approx(0.03)


def test_missing_features_filled_with_zero(monkeypatch):
    """モデルが期待する特徴量が不足していれば 0 で埋めて渡すこと。"""
    model = _make_model(0.01, expected_features=["a", "b", "c"])
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model", lambda name: model
    )

    run_inference(_request(features={"a": 1.0, "b": 2.0}))

    passed_df = model.predict.call_args[0][0]
    assert list(passed_df.columns) == ["a", "b", "c"]
    assert passed_df["c"].iloc[0] == 0


def test_all_models_fail_returns_zero_count(monkeypatch):
    """全モデルがロードできない場合 model_count=0 を返す（例外にしない）。"""
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model", lambda name: None
    )

    resp = run_inference(_request())

    assert resp.model_count == 0
    assert resp.used_models == []
    assert resp.avg_pred_price == 0.0
    assert resp.diff_ratio == 0.0


def test_partial_failure_renormalizes_weights(monkeypatch):
    """一部モデルが失敗した場合、成功分だけで重みを再正規化すること。"""
    model_b = _make_model(0.04, expected_features=["a", "b"])

    def _loader(name):
        return None if name == "ModelA" else model_b

    monkeypatch.setattr("services.prediction_service.inference.load_model", _loader)

    resp = run_inference(
        _request(model_types=["ModelA", "ModelB"], model_weights=[0.7, 0.3])
    )

    # ModelB のみ成功 → 重みは 1.0 に再正規化され、+0.04 がそのまま反映される
    assert resp.model_count == 1
    assert resp.used_models == ["ModelB"]
    assert resp.avg_pred_price == pytest.approx(1040.0)


def test_model_raising_exception_is_skipped(monkeypatch):
    """推論中に例外を投げるモデルはスキップして継続すること。"""
    bad = _make_model(0.0, expected_features=["a", "b"])
    bad.predict.side_effect = RuntimeError("boom")
    good = _make_model(0.02, expected_features=["a", "b"])

    def _loader(name):
        return bad if name == "ModelA" else good

    monkeypatch.setattr("services.prediction_service.inference.load_model", _loader)

    resp = run_inference(
        _request(model_types=["ModelA", "ModelB"], model_weights=[0.5, 0.5])
    )

    assert resp.model_count == 1
    assert resp.used_models == ["ModelB"]
    assert resp.avg_pred_price == pytest.approx(1020.0)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd python && py -m pytest tests/unit/services/test_inference.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'services.prediction_service.inference'`）

- [ ] **Step 3: 推論ロジックを実装**

`python/services/prediction_service/inference.py`:

```python
"""推論ロジック。

モデルのロード・特徴量アラインメント・推論・重み付きアンサンブルを行う。
DB にも yfinance にもアクセスしない（純粋な計算のみ）。

特徴量アラインメント（モデルの feature_names_in_ に合わせて不足分を 0 で埋め、
順序を揃える）はこのサービスの責務である。モデルファイルを持ち、期待される
特徴量名を知っているのがサービス側だからである。
"""

from __future__ import annotations

import logging
import os
from threading import Lock
from typing import Any

import joblib
import pandas as pd

from services.prediction_service.types import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# モデルディレクトリ。本体の python/models/unified/ をマウントして共有する。
MODEL_DIR = os.getenv("PREDICTION_MODEL_DIR", "/app/models/unified")

_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = Lock()


def clear_model_cache() -> None:
    """モデルキャッシュを空にする（テスト用）。"""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


def load_model(model_name: str) -> Any | None:
    """モデルをロードしてキャッシュする。見つからない場合は None を返す。"""
    with _CACHE_LOCK:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]

        path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
        try:
            model = joblib.load(path)
        except FileNotFoundError:
            logger.warning("モデルファイルが見つかりません: %s", path)
            model = None
        except Exception:
            logger.error("モデルロード失敗: %s", path, exc_info=True)
            model = None

        _MODEL_CACHE[model_name] = model
        return model


def _align_features(features: dict[str, float], model: Any) -> pd.DataFrame:
    """モデルが期待する特徴量に合わせて DataFrame を組み立てる。

    不足している特徴量は 0 で埋め、期待される順序に並べ替える。
    モデルが feature_names_in_ を持たない場合は与えられた特徴量をそのまま使う。
    """
    df = pd.DataFrame([features])

    inner = getattr(model, "model", None)
    expected = getattr(inner, "feature_names_in_", None)
    if expected is None:
        return df

    expected_list = list(expected)
    for feat in expected_list:
        if feat not in df.columns:
            df[feat] = 0
    return df[expected_list]


def _extract_prediction(pred: Any) -> float:
    """モデルの返り値（Series / list / スカラー）から float を取り出す。"""
    if isinstance(pred, pd.Series):
        return float(pred.iloc[-1])
    if isinstance(pred, (list, tuple)):
        return float(pred[-1])
    return float(pred)


def run_inference(request: PredictRequest) -> PredictResponse:
    """推論を実行し、重み付きアンサンブルの結果を返す。

    全モデルが失敗した場合も例外にせず、model_count=0 のレスポンスを返す
    （本体側はこれを「予測なし」として扱う）。
    一部のモデルのみ成功した場合、重みは成功分だけで再正規化する。
    """
    pred_prices: list[float] = []
    used_weights: list[float] = []
    used_models: list[str] = []

    for model_name, weight in zip(request.model_types, request.model_weights):
        model = load_model(model_name)
        if model is None:
            continue

        try:
            aligned = _align_features(request.features, model)
            pred_return = _extract_prediction(model.predict(aligned))
        except Exception:
            logger.warning(
                "モデル推論スキップ: model=%s market=%s symbol=%s",
                model_name,
                request.market,
                request.symbol,
                exc_info=True,
            )
            continue

        pred_prices.append(request.current_price * (1 + pred_return))
        used_weights.append(weight)
        used_models.append(model_name)

    if not pred_prices:
        return PredictResponse(
            market=request.market,
            symbol=request.symbol,
            current_price=request.current_price,
            avg_pred_price=0.0,
            diff_ratio=0.0,
            model_count=0,
            used_models=[],
        )

    # 成功したモデルのみで重みを再正規化する（合計が1になるよう保つ）
    weight_sum = sum(used_weights)
    if weight_sum > 0:
        normalized = [w / weight_sum for w in used_weights]
    else:
        normalized = [1.0 / len(pred_prices)] * len(pred_prices)

    avg_pred_price = sum(p * w for p, w in zip(pred_prices, normalized))
    diff_ratio = (avg_pred_price - request.current_price) / request.current_price

    return PredictResponse(
        market=request.market,
        symbol=request.symbol,
        current_price=request.current_price,
        avg_pred_price=float(avg_pred_price),
        diff_ratio=float(diff_ratio),
        model_count=len(pred_prices),
        used_models=used_models,
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `cd python && py -m pytest tests/unit/services/test_inference.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: Lint と型チェックを通す**

Run: `cd python && py -m black services/ tests/unit/services/ && py -m isort services/ tests/unit/services/ && py -m flake8 services/ tests/unit/services/ && py -m mypy services/`
Expected: エラーなし

- [ ] **Step 6: コミット**

```bash
git add python/services/prediction_service/inference.py python/tests/unit/services/test_inference.py
git commit -m "feat: 予測配信サービスの推論ロジックを追加"
```

---

### Task 3: FastAPI アプリ（app.py）

**Files:**
- Create: `python/services/prediction_service/app.py`
- Test: `python/tests/unit/services/test_app.py`

**Interfaces:**
- Consumes: Task 1 の型、Task 2 の `run_inference`, `load_model`
- Produces:
  - `app: FastAPI` — ASGI アプリケーション
  - `POST /predict` → `PredictResponse`
  - `GET /health` → `HealthResponse`

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/services/test_app.py`:

```python
"""FastAPI エンドポイントのテスト（TestClient 経由）。"""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from services.prediction_service.app import app
from services.prediction_service.inference import clear_model_cache


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def client():
    return TestClient(app)


def _make_model(pred_return: float):
    model = MagicMock()
    inner = MagicMock()
    inner.feature_names_in_ = ["a", "b"]
    model.model = inner
    model.predict.return_value = pd.Series([pred_return])
    return model


def _payload(**overrides) -> dict:
    body = {
        "market": "jp",
        "symbol": "7203",
        "current_price": 1000.0,
        "features": {"a": 1.0, "b": 2.0},
        "model_types": ["ModelA"],
        "model_weights": [1.0],
    }
    body.update(overrides)
    return body


def test_predict_returns_prediction(client, monkeypatch):
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model",
        lambda name: _make_model(0.02),
    )

    resp = client.post("/predict", json=_payload())

    assert resp.status_code == 200
    body = resp.json()
    assert body["market"] == "jp"
    assert body["symbol"] == "7203"
    assert body["model_count"] == 1
    assert body["avg_pred_price"] == pytest.approx(1020.0)


def test_predict_all_models_missing_returns_200_with_zero_count(client, monkeypatch):
    """全モデル失敗でも 200 を返し model_count=0 で表現すること。"""
    monkeypatch.setattr(
        "services.prediction_service.inference.load_model", lambda name: None
    )

    resp = client.post("/predict", json=_payload())

    assert resp.status_code == 200
    assert resp.json()["model_count"] == 0


def test_predict_weight_length_mismatch_returns_422(client):
    resp = client.post(
        "/predict", json=_payload(model_types=["A", "B"], model_weights=[1.0])
    )
    assert resp.status_code == 422


def test_predict_missing_required_field_returns_422(client):
    body = _payload()
    del body["current_price"]
    resp = client.post("/predict", json=body)
    assert resp.status_code == 422


def test_health_ok_when_model_loads(client, monkeypatch):
    monkeypatch.setattr(
        "services.prediction_service.app.load_model", lambda name: _make_model(0.0)
    )

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "UnifiedStockXGBoost" in body["loaded_models"]


def test_health_degraded_when_no_model_loads(client, monkeypatch):
    monkeypatch.setattr(
        "services.prediction_service.app.load_model", lambda name: None
    )

    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["loaded_models"] == []
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd python && py -m pytest tests/unit/services/test_app.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'services.prediction_service.app'`）

- [ ] **Step 3: FastAPI アプリを実装**

`python/services/prediction_service/app.py`:

```python
"""予測配信マイクロサービスの FastAPI アプリ。

起動:
    cd python
    uvicorn services.prediction_service.app:app --host 0.0.0.0 --port 5200
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from services.prediction_service.inference import load_model, run_inference
from services.prediction_service.types import HealthResponse, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# ヘルスチェックで存在確認する既定のモデル名
_DEFAULT_MODEL_NAMES = ("UnifiedStockXGBoost", "UnifiedStockLightGBM")

app = FastAPI(
    title="StockFixer Prediction Service",
    description="特徴量と現在価格を受け取り、モデル推論とアンサンブルのみを行う計算サービス",
    version="1.0.0",
)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """推論を実行して予測価格を返す。

    全モデルの推論が失敗した場合も 200 を返し、model_count=0 で表現する。
    """
    return run_inference(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """モデルがロード可能かを返す。

    1つもロードできない場合は status="degraded" とする（HTTP は 200 のまま。
    サービスプロセス自体は生きているため）。
    """
    loaded = [name for name in _DEFAULT_MODEL_NAMES if load_model(name) is not None]
    return HealthResponse(status="ok" if loaded else "degraded", loaded_models=loaded)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `cd python && py -m pytest tests/unit/services/test_app.py -v`
Expected: PASS（6 tests）

- [ ] **Step 5: Lint と型チェックを通す**

Run: `cd python && py -m black services/ tests/unit/services/ && py -m isort services/ tests/unit/services/ && py -m flake8 services/ tests/unit/services/ && py -m mypy services/`
Expected: エラーなし

- [ ] **Step 6: コミット**

```bash
git add python/services/prediction_service/app.py python/tests/unit/services/test_app.py
git commit -m "feat: 予測配信サービスのFastAPIエンドポイントを追加"
```

---

### Task 4: 本体側クライアント（remote_client.py）

**Files:**
- Create: `python/src/prediction/remote_client.py`
- Test: `python/tests/unit/test_prediction_remote_client.py`

**Interfaces:**
- Consumes: なし（HTTP 経由なのでサービス側コードを import しない）
- Produces:
  - `get_service_url() -> str | None` — `PREDICTION_SERVICE_URL` を返す（未設定なら None）
  - `predict_via_service(market, symbol, current_price, features, model_types, model_weights) -> PredictionResult | None` — サービスを呼ぶ。呼べない/失敗した場合は `None`（呼び出し側がフォールバックする合図）

**重要:** このモジュールは `services/` を import してはならない（本体とサービスは独立してデプロイされるため、HTTP のみが契約である）。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_prediction_remote_client.py`:

```python
"""本体側の推論サービスクライアントのテスト。

サービス障害時に例外を投げず None を返す（＝インプロセス推論へフォールバック
する合図）ことを重点的に検証する。
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.prediction.remote_client import get_service_url, predict_via_service


def _call(**overrides):
    kwargs = {
        "market": "jp",
        "symbol": "7203",
        "current_price": 1000.0,
        "features": {"a": 1.0},
        "model_types": ["ModelA"],
        "model_weights": [1.0],
    }
    kwargs.update(overrides)
    return predict_via_service(**kwargs)


class TestGetServiceUrl:
    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("PREDICTION_SERVICE_URL", raising=False)
        assert get_service_url() is None

    def test_returns_url_when_set(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://localhost:5200")
        assert get_service_url() == "http://localhost:5200"

    def test_blank_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "   ")
        assert get_service_url() is None


class TestPredictViaService:
    def test_returns_none_when_url_unset(self, monkeypatch):
        """URL 未設定なら HTTP を呼ばずに None を返す（既定動作）。"""
        monkeypatch.delenv("PREDICTION_SERVICE_URL", raising=False)
        with patch("src.prediction.remote_client.requests.post") as mock_post:
            assert _call() is None
            mock_post.assert_not_called()

    def test_successful_response_mapped_to_prediction_result(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "market": "jp",
            "symbol": "7203",
            "current_price": 1000.0,
            "avg_pred_price": 1020.0,
            "diff_ratio": 0.02,
            "model_count": 2,
            "used_models": ["ModelA", "ModelB"],
        }

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            result = _call()

        assert result is not None
        assert result.market == "jp"
        assert result.symbol == "7203"
        assert result.avg_pred_price == pytest.approx(1020.0)
        assert result.diff_ratio == pytest.approx(0.02)
        assert result.model_count == 2

    def test_zero_model_count_returns_none(self, monkeypatch):
        """model_count=0 は「予測なし」を意味するので None を返す。"""
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "market": "jp",
            "symbol": "7203",
            "current_price": 1000.0,
            "avg_pred_price": 0.0,
            "diff_ratio": 0.0,
            "model_count": 0,
            "used_models": [],
        }

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            assert _call() is None

    def test_timeout_returns_none(self, monkeypatch):
        """タイムアウトは例外を伝播させず None を返す（フォールバックの合図）。"""
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        with patch(
            "src.prediction.remote_client.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            assert _call() is None

    def test_connection_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        with patch(
            "src.prediction.remote_client.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            assert _call() is None

    def test_server_error_returns_none(self, monkeypatch):
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            assert _call() is None

    def test_malformed_response_returns_none(self, monkeypatch):
        """必須キーが欠けた応答でも例外を投げず None を返すこと。"""
        monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unexpected": "shape"}

        with patch("src.prediction.remote_client.requests.post", return_value=mock_resp):
            assert _call() is None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd python && py -m pytest tests/unit/test_prediction_remote_client.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.prediction.remote_client'`）

- [ ] **Step 3: クライアントを実装**

`python/src/prediction/remote_client.py`:

```python
"""予測配信マイクロサービスの HTTP クライアント（#予測配信サービス フェーズ1）。

環境変数 PREDICTION_SERVICE_URL が設定されているときだけサービスを呼ぶ。
未設定時・呼び出し失敗時はいずれも None を返し、呼び出し側は従来の
インプロセス推論へフォールバックする。

このモジュールは services/ を import しない。本体とサービスは独立して
デプロイされるため、HTTP のみが両者の契約である。
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from src.prediction.types import PredictionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUEST_TIMEOUT_SECONDS = 10.0


def get_service_url() -> Optional[str]:
    """PREDICTION_SERVICE_URL を返す。未設定・空文字なら None。"""
    url = os.getenv("PREDICTION_SERVICE_URL", "").strip()
    return url or None


def predict_via_service(
    market: str,
    symbol: str,
    current_price: float,
    features: dict[str, float],
    model_types: list[str],
    model_weights: list[float],
) -> Optional[PredictionResult]:
    """推論サービスへ HTTP で問い合わせる。

    Returns:
        成功時は PredictionResult。以下の場合はいずれも None を返し、
        呼び出し側はインプロセス推論へフォールバックする:
          - PREDICTION_SERVICE_URL 未設定
          - 接続不可・タイムアウト・5xx
          - 応答が想定した形式でない
          - model_count が 0（サービス側で全モデル失敗＝予測なし）
    """
    base_url = get_service_url()
    if base_url is None:
        return None

    payload = {
        "market": market,
        "symbol": symbol,
        "current_price": current_price,
        "features": features,
        "model_types": model_types,
        "model_weights": model_weights,
    }

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/predict",
            json=payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.RequestException as exc:
        logger.warning(
            "推論サービス呼び出し失敗（インプロセス推論にフォールバック）: "
            "market=%s symbol=%s error=%s",
            market,
            symbol,
            exc,
        )
        return None

    if response.status_code != 200:
        logger.warning(
            "推論サービスがエラー応答（インプロセス推論にフォールバック）: "
            "market=%s symbol=%s status=%s",
            market,
            symbol,
            response.status_code,
        )
        return None

    try:
        body = response.json()
        model_count = int(body["model_count"])
        if model_count == 0:
            return None
        return PredictionResult(
            market=str(body["market"]),
            symbol=str(body["symbol"]),
            current_price=float(body["current_price"]),
            avg_pred_price=float(body["avg_pred_price"]),
            diff_ratio=float(body["diff_ratio"]),
            model_count=model_count,
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "推論サービスの応答を解釈できません（インプロセス推論にフォールバック）: "
            "market=%s symbol=%s error=%s",
            market,
            symbol,
            exc,
        )
        return None
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `cd python && py -m pytest tests/unit/test_prediction_remote_client.py -v`
Expected: PASS（10 tests）

- [ ] **Step 5: Lint・型チェック・アーキテクチャ契約を通す**

Run: `cd python && py -m black src/prediction/remote_client.py tests/unit/test_prediction_remote_client.py && py -m isort src/prediction/remote_client.py tests/unit/test_prediction_remote_client.py && py -m flake8 src/prediction/remote_client.py tests/unit/test_prediction_remote_client.py && py -m mypy src/prediction/remote_client.py && lint-imports`
Expected: すべてエラーなし（import-linter も通ること）

- [ ] **Step 6: コミット**

```bash
git add python/src/prediction/remote_client.py python/tests/unit/test_prediction_remote_client.py
git commit -m "feat: 推論サービス呼び出しクライアントを追加（既定OFF・フォールバック付き）"
```

---

### Task 5: 本体の予測経路への組み込み

**Files:**
- Modify: `python/src/prediction/predict_unified.py`（`predict_with_unified_model` 内）
- Test: `python/tests/unit/test_predict_unified_service_integration.py`

**Interfaces:**
- Consumes: Task 4 の `predict_via_service`
- Produces: なし（既存関数の挙動を拡張するのみ）

**方針:** `predict_with_unified_model()` の中で、特徴量読み込みと現在価格取得が終わった直後、モデル推論ループに入る前に `predict_via_service()` を試す。成功すればその結果を返し、`None` が返ればそのまま既存のインプロセス推論を続行する。

**重要 1（既定パスの性能）:** サービスに渡す `model_weights` は `load_model_weights()`（DB クエリ）で取得する必要があるが、これを無条件に呼ぶと `PREDICTION_SERVICE_URL` 未設定の既定パスでも銘柄ごとに余計な DB クエリが1回増えてしまう（792銘柄 × 1クエリ）。**必ず `get_service_url()` で有効性を確認してから**重みを計算すること。

**重要 2（重みの取り違え）:** 既存コードはループ後に `load_model_weights(market, symbol, succeeded_model_names)`（成功したモデルのみ）を呼んでいる。サービス用には `model_types`（全モデル）を渡すため意味が異なる。サービス用の重みは別変数（`service_weights`）で保持し、既存の呼び出しには手を触れないこと。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_predict_unified_service_integration.py`:

```python
"""predict_with_unified_model の推論サービス連携テスト。

サービスが使える場合はその結果を返し、使えない場合は従来のインプロセス推論に
フォールバックすることを検証する。
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.prediction.predict_unified import predict_with_unified_model
from src.prediction.types import PredictionResult


@pytest.fixture
def _feature_df():
    return pd.DataFrame(
        {
            "Close_lag1": [1000.0, 1010.0],
            "rsi_lag1": [50.0, 55.0],
            "y": [0.01, 0.02],
        }
    )


def _patch_common(feature_df):
    """特徴量読み込みと現在価格取得を固定するパッチ群を返す。"""
    return [
        patch("src.prediction.predict_unified.load_feature_data", return_value=feature_df),
        patch("src.prediction.predict_unified.load_model_weights", return_value=[0.5, 0.5]),
    ]


def _mock_model():
    model = MagicMock()
    inner = MagicMock()
    inner.feature_names_in_ = ["Close_lag1", "rsi_lag1"]
    model.model = inner
    model.predict.return_value = pd.Series([0.02])
    return model


def test_uses_service_result_when_available(_feature_df, monkeypatch):
    """サービスが結果を返した場合、インプロセス推論を行わずそれを返すこと。"""
    monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
    service_result = PredictionResult(
        market="jp",
        symbol="7203",
        current_price=1000.0,
        avg_pred_price=1050.0,
        diff_ratio=0.05,
        model_count=2,
    )

    patches = _patch_common(_feature_df)
    with patches[0], patches[1], patch(
        "src.prediction.predict_unified.predict_via_service", return_value=service_result
    ), patch("src.prediction.predict_unified.get_cached_model") as mock_get_model, patch(
        "src.prediction.predict_unified.yf.Ticker"
    ) as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    assert result is service_result
    mock_get_model.assert_not_called()


def test_falls_back_to_inprocess_when_service_returns_none(_feature_df, monkeypatch):
    """サービスが None を返した場合、従来のインプロセス推論が動くこと。"""
    monkeypatch.setenv("PREDICTION_SERVICE_URL", "http://svc:5200")
    model = _mock_model()

    patches = _patch_common(_feature_df)
    with patches[0], patches[1], patch(
        "src.prediction.predict_unified.predict_via_service", return_value=None
    ), patch(
        "src.prediction.predict_unified.get_cached_model", return_value=model
    ), patch("src.prediction.predict_unified.yf.Ticker") as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    assert result is not None
    assert result.model_count > 0
    assert model.predict.called


def test_service_not_called_when_url_unset(_feature_df, monkeypatch):
    """URL 未設定時はサービス呼び出しも重み計算(DBクエリ)も行わないこと。

    既定パスで銘柄ごとに余計な DB クエリが増えることを防ぐための回帰テスト。
    """
    monkeypatch.delenv("PREDICTION_SERVICE_URL", raising=False)
    model = _mock_model()

    with patch(
        "src.prediction.predict_unified.load_feature_data", return_value=_feature_df
    ), patch(
        "src.prediction.predict_unified.load_model_weights", return_value=[1.0]
    ) as mock_weights, patch(
        "src.prediction.predict_unified.predict_via_service"
    ) as mock_service, patch(
        "src.prediction.predict_unified.get_cached_model", return_value=model
    ), patch(
        "src.prediction.predict_unified.yf.Ticker"
    ) as mock_ticker:
        mock_ticker.return_value.history.return_value = pd.DataFrame({"Close": [1000.0]})

        result = predict_with_unified_model("jp", "7203")

    mock_service.assert_not_called()
    # 既存経路のループ後の1回だけ（サービス用の事前計算が走っていないこと）
    assert mock_weights.call_count == 1
    assert result is not None
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd python && py -m pytest tests/unit/test_predict_unified_service_integration.py -v`
Expected: FAIL（`predict_via_service` が `predict_unified` にまだ存在しないため `AttributeError`）

- [ ] **Step 3: predict_unified.py に import を追加**

`python/src/prediction/predict_unified.py` の import 部（15行目付近、`from src.prediction.db import load_model_weights` の直後）に追加:

```python
from src.prediction.remote_client import get_service_url, predict_via_service
```

- [ ] **Step 4: サービス呼び出しを組み込む**

`predict_with_unified_model()` の中、`market_encoded` 列を追加している箇所（159-162行目付近）の直後、`# 各モデルで予測（キャッシュされたモデルを使用）` コメントの直前に以下を挿入する:

```python
    # 推論サービスが有効なときだけ委譲する。
    # get_service_url() で先に判定するのは、未設定の既定パスで load_model_weights()
    # の DB クエリを余計に走らせないため（銘柄数ぶん積み上がるため無視できない）。
    if get_service_url() is not None:
        service_weights = load_model_weights(market, symbol, model_types)
        service_result = predict_via_service(
            market=market,
            symbol=symbol,
            current_price=float(current_price),
            features={
                str(col): float(latest_X[col].iloc[0])
                for col in latest_X.columns
                if pd.notna(latest_X[col].iloc[0])
            },
            model_types=list(model_types),
            model_weights=service_weights,
        )
        # None のときは下のインプロセス推論にそのままフォールバックする
        if service_result is not None:
            return service_result

```

- [ ] **Step 5: テストを実行して成功を確認**

Run: `cd python && py -m pytest tests/unit/test_predict_unified_service_integration.py -v`
Expected: PASS（3 tests）

- [ ] **Step 6: 既存の予測テストが壊れていないことを確認**

Run: `cd python && py -m pytest tests/unit/ -k "predict or unified" -v`
Expected: すべて PASS（既存テストは `PREDICTION_SERVICE_URL` 未設定なので挙動不変）

- [ ] **Step 7: Lint・型チェックを通す**

Run: `cd python && py -m black src/prediction/predict_unified.py tests/unit/test_predict_unified_service_integration.py && py -m isort src/prediction/predict_unified.py tests/unit/test_predict_unified_service_integration.py && py -m flake8 src/prediction/predict_unified.py tests/unit/test_predict_unified_service_integration.py && py -m mypy src/prediction/predict_unified.py`
Expected: エラーなし

- [ ] **Step 8: コミット**

```bash
git add python/src/prediction/predict_unified.py python/tests/unit/test_predict_unified_service_integration.py
git commit -m "feat: 予測経路に推論サービス委譲を組み込み（未設定時は従来通り）"
```

---

### Task 6: Dockerfile・README・バージョン更新

**Files:**
- Create: `python/services/Dockerfile.prediction`
- Create: `python/services/prediction_service/README.md`
- Modify: `python/VERSION`

**Interfaces:**
- Consumes: Task 1-5 の成果物すべて
- Produces: なし（ドキュメントとビルド定義のみ）

- [ ] **Step 1: Dockerfile を作成**

`python/services/Dockerfile.prediction`:

```dockerfile
# 予測配信マイクロサービス用イメージ（本体 stockfixer とは別イメージ）。
#
# ビルド（python/ ディレクトリから）:
#   docker build -f services/Dockerfile.prediction -t stockfixer-prediction:dev .
#
# 起動:
#   docker run -p 5200:5200 -v "$(pwd)/models:/app/models:ro" stockfixer-prediction:dev
FROM python:3.12-slim

WORKDIR /app

# 推論に必要な依存のみをインストールする。
# requirements.txt（本体用）は入れない — サービスは DB も yfinance も使わないため。
# ML ライブラリのバージョンは requirements-service.txt 側で本体と同じピンに
# 揃えてある（joblib モデルの互換性のため）。ここには直書きしない。
COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

COPY services/ ./services/

ENV PREDICTION_MODEL_DIR=/app/models/unified
EXPOSE 5200

CMD ["uvicorn", "services.prediction_service.app:app", "--host", "0.0.0.0", "--port", "5200"]
```

- [ ] **Step 2: README を作成**

`python/services/prediction_service/README.md`:

```markdown
# 予測配信マイクロサービス（学習用フェーズ1）

特徴量と現在価格を受け取り、モデル推論とアンサンブルのみを行う純粋な計算サービス。

**設計書:** `docs/superpowers/specs/2026-07-31-prediction-microservice-design.md`

## 設計方針

- **DB 接続を持たない** — Postgres の接続情報を一切持たず、特徴量もモデル重みも
  呼び出し側（本体）が HTTP ペイロードとして渡す
- **yfinance を呼ばない** — 現在価格の取得は本体の責務
- **読むのはモデルファイルのみ** — `python/models/unified/*.joblib` を read-only で参照

この境界により、サービスは外部 I/O ゼロとなりテストが容易になる。

## ローカル起動

```bash
cd python
pip install -r requirements-service.txt
uvicorn services.prediction_service.app:app --host 0.0.0.0 --port 5200
```

Docker で起動する場合:

```bash
cd python
docker build -f services/Dockerfile.prediction -t stockfixer-prediction:dev .
docker run -p 5200:5200 -v "$(pwd)/models:/app/models:ro" stockfixer-prediction:dev
```

## 動作確認

```bash
curl http://localhost:5200/health

curl -X POST http://localhost:5200/predict \
  -H "Content-Type: application/json" \
  -d '{
    "market": "jp",
    "symbol": "7203",
    "current_price": 2500.0,
    "features": {"Close_lag1": 2480.0},
    "model_types": ["UnifiedStockXGBoost"],
    "model_weights": [1.0]
  }'
```

## 本体からの利用

既定では**無効**。環境変数 `PREDICTION_SERVICE_URL` を設定したときだけ本体が
このサービスを呼ぶ。

```bash
export PREDICTION_SERVICE_URL=http://localhost:5200
```

サービスが停止している・タイムアウトした・5xx を返した場合、本体は警告ログを
出して従来のインプロセス推論にフォールバックするため、本番を壊すことはない。

## エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| POST | `/predict` | 推論を実行。全モデル失敗時も 200 で `model_count: 0` を返す |
| GET | `/health` | モデルがロード可能かを返す |

## 環境変数

| 変数 | 既定値 | 説明 |
|---|---|---|
| `PREDICTION_MODEL_DIR` | `/app/models/unified` | joblib モデルの配置ディレクトリ |
```

- [ ] **Step 3: VERSION を更新**

`python/VERSION` の内容を現在の値から minor を1つ上げた値に変更する。

Run: `cat python/VERSION` で現在値を確認し、`X.Y.Z` → `X.(Y+1).0` に更新する。
（例: `2.1.7` なら `2.2.0`）

新機能追加のため minor bump とする（`docs/VERSIONING_POLICY.md` 準拠）。

- [ ] **Step 4: 全体テストとCI相当チェックを実行**

Run: `cd python && py -m pytest tests/unit/ --cov=src --cov-branch --cov-fail-under=80 -q`
Expected: すべて PASS、カバレッジ 80% 以上

Run: `cd python && py -m black --check services/ src/ tests/ && py -m isort --check-only services/ src/ tests/ && py -m flake8 services/ src/ tests/ && py -m mypy src/ services/ && lint-imports`
Expected: すべてエラーなし

- [ ] **Step 5: コミット**

```bash
git add python/services/Dockerfile.prediction python/services/prediction_service/README.md python/VERSION
git commit -m "docs: 予測配信サービスのDockerfileとREADMEを追加"
```

---

## 完了条件

- [ ] 推論サービスを `uvicorn` で起動でき、`curl` で `/health` と `/predict` が応答する
- [ ] `PREDICTION_SERVICE_URL` 未設定時、既存の全テストが変わらず通る
- [ ] サービス障害時（未起動・タイムアウト・5xx）にフォールバックが働くことがテストで検証されている
- [ ] `lint-imports` が通る（`services/` が `src/` のレイヤー契約を汚していない）
- [ ] カバレッジ 80% 以上を維持
