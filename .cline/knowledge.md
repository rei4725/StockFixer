# Tips

## Discord Bot `/Next10` コマンド運用Tips
- `/Next10` コマンドで python/results/top10_diff_stocks.csv を参照し、内容をDiscordへ送信する実装例あり
- 計算処理は外部スクリプトで事前実行し、BotはCSVのみを参照することで応答高速化
- CSVの絶対パス参照でパス不整合を防止
- 出力先ディレクトリ変更時はスクリプト・Bot両方のパス修正が必要
- Discordメッセージ長制限（2000文字）に注意し、分割送信処理を実装

## Discord連携Tips
- PythonからDiscord通知を行う場合はWebhookを利用するのが簡便
- Webhook URLは環境変数で管理し、ハードコーディングしない
- 送信はrequests.postでOK、成功時は204/200、失敗時は例外処理で握りつぶさずログ出力
- 重要通知はtry-exceptでAPI本体の処理と分離することで障害伝播を防止

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

## バックテスト・パスエラー対応Tips
- run_backtest_sample.py実行時、sys.path.appendで絶対パス/相対パスを明示的に追加すること
- data_loader.pyのimportはfrom data_loader import ...形式で直接指定
- python/src/data/配下のimport時はsys.pathにpython/src/dataを追加
- モジュール名・ファイル名の重複やパスの曖昧さに注意
- 特徴量CSVにDate列がない場合はparse_dates, index_col指定を外す
- テスト・スクリプトのパス設計を統一し、importエラーを防止
