新しい銘柄を監視対象に追加し、データ取得・モデル学習を完了する。

## 手順

### 1. ウォッチリストに追加
`python/データ取得対象.csv` に `market,symbol` 行を追加する。
```csv
jp,9984
us,MSFT
```
- 日本株: market=`jp`, symbol=数字のみ（`.T` サフィックスは `get_ticker()` で自動補正）
- 米国株: market=`us`, symbol=ティッカーシンボル

### 2. データ取得
```bash
cd python
py run_data_creation.py --market jp --symbol 9984
```
過去5年分のOHLCVを取得し、特徴量を生成してDBに保存する。

### 3. モデル学習（銘柄別モデルが必要な場合のみ）
```bash
py run_model_creation.py --market jp --symbol 9984
```
統合モデル使用の場合はこの手順は不要（次回の週次再学習で反映）。

### 4. 確認
```bash
py python/tools/check_data.py
```
DBに正しくデータが保存されていることを確認する。

## 注意事項
- 次回のスケジューラー実行（`daily_pipeline`）で自動的に予測対象に含まれる
- 統合モデルの精度向上には週次再学習（`weekly_model_training`）後に反映
- 上場廃止銘柄を追加すると yfinance がエラーを返す可能性がある
