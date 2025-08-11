# Tips

## ファイル・ディレクトリ運用
- 命名は一貫性を重視し、スネークケースを推奨（例: user_profile.md）
- 拡張子は用途ごとに.md, .json, .yaml等を使い分ける
- サブディレクトリを極力排除し、主要.pyファイルは1階層に集約
- importパスも1階層構成に統一し、修正やリファクタ時の影響範囲を明確化
- `[market]_[symbol]` サブディレクトリ＋用途明示ファイル名で混在・誤保存・誤読込を防止
- 生データは `python/data/`、ロジックは `python/src/data/logic/` に集約
- モデル保存は `python/models/[market]_[symbol]/モデル名.joblib` 形式で管理

## Python開発・運用
- Pythonコマンドは「py」を推奨、仮想環境は `py -m venv .venv`
- ライブラリインストールは `cmd /c` 経由で実行
- Windowsではパス区切りに「\」を使用
- モジュール化で再利用性・可読性向上、`__init__.py` 配置でimportエラー防止
- インポートエラー時はパス・環境変数・仮想環境・pip install状況を確認
- unittest.mockやMagicMockで外部依存をモック

## データ取得・処理
- yfinanceで株価・為替レート等を安定取得、ccxtは暗号資産取引所API用
- yfinanceのMultiIndexカラムはフラット化して後続処理を簡素化
- バッチ処理はmarket/symbolのみ渡し、既存csv全削除で一貫性担保
- 特徴量生成は全数値列にラグ特徴量・テクニカル指標を自動付与

## モデル運用
- 複数モデルの予測値は平均してバイアス低減
- 並列処理は競合バグの原因となるため同期集計が安定
- モデルファイルはjoblib形式で保存し、パス問題はPYTHONPATH等で回避
- 期間指定はend_dateを現在日時、start_dateを5年前に自動設定
- 予測値は直近データの翌営業日終値

## 自動化・スクレイピング
- Seleniumは対応ブラウザのWebDriver（例: ChromeDriver）が必要
- WebDriverはPATH追加またはコードでパス明示、ヘッドレスモード推奨
- 要素待機はWebDriverWait/expected_conditionsを利用
- サイト構造変更に注意し、要素ID等は定期確認

---

# 対応必須内容

- モジュール・パス・環境変数等のエラー対策
- データ保存前のcsv全削除による一貫性担保
- サブディレクトリ・ファイル名の一貫性維持
- モデル保存時のmarket,symbol抽出・パス設計
- テストやスクリプトのパス修正

---
