"""永続化アダプタ — src/domain/ports.py のリポジトリ系ポートの Postgres 実装。

テーブルごとにモジュールを分ける。BC はここを import せず、合成ルート
（run_*.py / src/orchestration）が構築して BC へ注入する。
"""
