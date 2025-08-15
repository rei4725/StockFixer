## 指示：バックテストの実行と修正
- 原文：バックテストの実行と、結果を確認してください。修正が必要であれば修正してください
- 要約：バックテストスクリプトの実行とエラー修正
- 対応状況：完了
- 関連タスク：バックテスト実行・修正

## 指示：SignalGeneratorテスト修正
- 原文：TestSignalGenerator の結果が成功するように変更して
- 要約：signal_generator.pyのロジック修正・テスト全件成功
- 対応状況：完了
- 関連タスク：バックテスト実行・修正

## 指示：ModelManagerテスト修正
- 原文：test_register_model_type が成功するようにしてください。もしくは、ケース不要であれば削除してください
- 要約：register_model_type仕様確認・テストパス修正・全件成功
- 対応状況：完了
- 関連タスク：ModelManagerテスト・仕様確認

## 指示：update cline
- 原文：update cline
- 要約：.cline配下の管理ファイルを現状に合わせて更新
- 対応状況：対応済み
- 関連タスク：バックテスト実行・修正

## 指示：discord_bot.pyの/Next10コマンド機能改修
- 原文：'python\src\api\discord_bot.py' について、機能改修をおこなって。・メッセージの内容が「/Next10」だった場合、'python/run_top10_diff_stocks.py' の実行結果から、　df_result 変数 をメッセージとして送信するような機能
- 要約：/Next10コマンドでpython/results/top10_diff_stocks.csvを参照しDiscordへ送信する機能を追加
- 対応状況：対応済み
- 関連タスク：discord_bot.pyの/Next10コマンド機能改修

## 指示：discord出力仕様統一・部品化
- 原文：dfの中身は全く同じであるとして問題ない、列操作の部分も共通化して、「 シンボル   現在値  予想終値 予想変化率 」が出力される形に統一して
- 要約：convert_df_for_discord関数でdf変換・出力仕様を統一
- 対応状況：対応済み
- 関連タスク：discord出力仕様統一・部品化

## 指示：get_stock_data引数追加・関連修正
- 原文：get_stock_data の引数追加に伴い、修正が必要な箇所を修正したいです。判断が必要な箇所以外は機械的に作業してください
- 要約：market引数追加、呼び出し箇所の機械的修正、テスト修正、API/モデル/データ取得系の対応
- 対応状況：進行中
- 関連タスク：get_stock_data引数追加・関連修正

## 指示：predict_single_stock自動モデル生成・日本株対応
- 原文：モデル未存在時も一連処理を自動化、日本株（トヨタ）で動作確認、ティッカー補正修正
- 要約：モデル未存在時の自動データ取得・csv保存・モデル作成・学習・保存・予測、日本株対応
- 対応状況：対応済み
- 関連タスク：predict_single_stock自動モデル生成・日本株対応

## 指示：マーケット別Top10・ワースト10出力＆DiscordBot連携
- 原文：run_top10_diff_stocks.pyをマーケット毎にランキング出力、ワースト10も出力。discord_bot.pyのコマンドを/forecastに変更し、全マーケットのTop10・ワースト10を送信する仕様にしてください
- 要約：run_top10_diff_stocks.pyをマーケット毎にTop10・ワースト10抽出＆サブフォルダ保存に改修。discord_bot.pyを/forecastコマンドで全マーケット結果送信仕様に改修
- 対応状況：対応済み
- 関連タスク：マーケット別Top10・ワースト10出力＆DiscordBot連携

## 指示：パス・ティッカー補正のutils化
- 原文：ファイルパス・ティッカー補正処理をutilsに切り出し、各所で利用するようリファクタリング
- 要約：data_path_utils.pyにget_data_subdir, get_models_subdir, get_ticker等を実装し、各データ・モデル操作で利用
- 対応状況：対応済み
- 関連タスク：パス・ティッカー補正のutils化
