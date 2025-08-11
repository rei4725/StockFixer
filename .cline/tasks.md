## タスク名：株価データ特徴量拡張
- 状況：完了
- 優先度：高
- 期限：2025-08-11
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - テクニカル指標（RSI, MACD, EMA, ATR等）を追加し、全数値列のラグ特徴量を自動生成
  - data_saver.py, technical_analysis.py を修正

---

## タスク名：run_data_creation.py例外防止・yfinance仕様対応
- 状況：完了
- 優先度：高
- 期限：2025-08-11
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - yfinanceのマルチインデックス仕様に対応し、カラム名をフラット化
  - 例外発生箇所を修正し、正常終了を確認

---

<!-- 他タスクは省略 -->

---

## タスク名：モデル保存ディレクトリ分割・再帰的データ取得対応
- 状況：完了
- 優先度：高
- 期限：2025-08-11
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - モデル保存先を python/models/[market]_[symbol]/モデル名.joblib 形式に変更
  - python/data配下のサブディレクトリも含めてcsvファイルを再帰的に探索
  - サブディレクトリ名からmarket, symbolを抽出し、個別にモデル作成・保存
  - run_model_creation.py, model_manager.py を修正


## タスク名：出力ファイルパス・ファイル名設計改善
- 状況：完了
- 優先度：中
- 期限：2025-08-11
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - [market]_[symbol] サブディレクトリを自動生成し、その中にcsvを保存
  - ファイル名を features_YYYY_MM_DD_YYYY_MM_DD.csv 形式に統一
  - 既存run_data_creation.py等の呼び出し方法は変更不要

---

## タスク名：バッチデータ生成自動化・既存csv削除
- 状況：完了
- 優先度：高
- 期限：2025-08-11
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - batch_run_data_creation.pyからrun_data_creation.pyを呼び出し、market/symbol単位で自動実行
  - save_stock_data_with_featuresでstart_date, end_date自動決定
  - データ保存前に同一ディレクトリ内の既存csvファイルを全削除

### 更新履歴
- 2025-08-11 16:58：株価データ特徴量拡張タスクを追加・完了で記載
- 2025-08-11 17:43：run_data_creation.py例外防止・yfinance仕様対応タスクを追加・完了で記載
- 2025-08-11 18:31：バッチデータ生成自動化・既存csv削除タスクを追加・完了で記載
