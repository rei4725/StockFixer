## タスク名：バックテスト実行・修正
- 状況：進行中
- 優先度：高
- 期限：2025-08-11
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - run_backtest_sample.pyの実行でモジュールパスエラー発生
  - data_loader.pyのDate列対応修正済み
  - sys.path, import文の修正が必要
  - テスト・スクリプトのパス設計見直し要

## タスク名：Discord連携機能の実装
- 状況：完了
- 優先度：中
- 期限：2025-08-13
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - python/src/api/discord_webhook.py新規作成
  - api_server.pyの/generate_signalでDiscord通知を実装
  - Webhook URLは環境変数で管理

## タスク名：discord_bot.pyの/Next10コマンド機能改修
- 状況：完了
- 優先度：高
- 期限：2025-08-13
- 関連ファイル：context.md, instructions.md
- メモ：
  - /Next10コマンドでpython/results/top10_diff_stocks.csvを参照しDiscordへ送信
  - 出力先ディレクトリ・パスの修正対応済み
  - 計算処理は外部スクリプトで事前実行

## タスク名：discord出力仕様統一・部品化
- 状況：完了
- 優先度：高
- 期限：2025-08-14
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - df変換部品化（convert_df_for_discord）
  - 出力仕様「シンボル 現在値 予想終値 予想変化率」に統一
  - テスト検証済み

## タスク名：get_stock_data引数追加・関連修正
- 状況：進行中
- 優先度：高
- 期限：2025-08-14
- 関連ファイル：context.md, instructions.md, knowledge.md
- メモ：
  - market引数追加
  - 呼び出し箇所の機械的修正（data_saver.py, data_loader.py, predict_single_stock.py, api_server.py, test_data_loader.py）
  - テスト修正・API/モデル/データ取得系の対応
  - 既存機能への影響確認
