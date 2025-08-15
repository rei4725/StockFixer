## プロジェクト：StockFixer - 株式自動売買システム
- 目的：Pythonによる戦略ロジックとAI予測、C#による注文実行とUI表示を組み合わせたリアルタイム自動売買システムを構築し、利益を最大化する。
- 制約：テスト仕様に合わせてSignalGenerator・ModelManagerのロジック整理
- 関連：
  - PROJECT_OVERVIEW.md を参照する
  - .cline/knowledge.md テスト運用Tips参照

## 背景：マーケット別ランキング・DiscordBot連携
- run_top10_diff_stocks.pyでマーケット毎にTop10・ワースト10を抽出し、実行日時サブフォルダにmarket別でCSV保存
- DiscordBotは/forecastコマンドで最新結果フォルダから全market分をヘッダー付きで送信
- 2000文字超のメッセージは分割送信でDiscord制限に対応
- ファイル命名・サブフォルダ設計・DataFrame表示仕様は一貫性を重視
