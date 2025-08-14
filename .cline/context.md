## プロジェクト：CuteStock - 株式自動売買システム
- 目的：Pythonによる戦略ロジックとAI予測、C#による注文実行とUI表示を組み合わせたリアルタイム自動売買システムを構築し、利益を最大化する。
- 制約：
- 関連：
  - PROJECT_OVERVIEW.md を参照する

## プロジェクト：Discord Bot `/Next10` コマンド機能
- 目的：Discord上で `/Next10` コマンドを受信した際、python/results/top10_diff_stocks.csv の内容を即時返信し、株式AI予測の上位10銘柄情報を共有する
- 制約：計算処理は外部スクリプトで事前実行、BotはCSVのみ参照
- 関連：python/src/api/discord_bot.py, python/run_top10_diff_stocks.py, python/results/top10_diff_stocks.csv

## プロジェクト：Discord出力仕様統一・部品化
- 目的：Discord Botの出力仕様（銘柄テーブル）を部品化し、複数機能で統一的に利用できるようにする
- 制約：dfの中身・列構成は完全に統一、convert_df_for_discord関数で一元管理
- 関連：python/src/api/discord_bot.py, .cline/knowledge.md, .cline/instructions.md
