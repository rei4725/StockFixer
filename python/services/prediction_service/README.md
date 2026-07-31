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
