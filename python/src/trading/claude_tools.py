"""
ClaudeTrader に渡す tool のスキーマ定義とシステムプロンプト

claude_agent.py から分離（ファイル行数ゲート対応）。
"""

from __future__ import annotations

from typing import Any

from config.settings import MAX_ORDERS_PER_RUN


def build_system_prompt(gate_notice: str, market: str, mode: str) -> str:
    """ClaudeTrader のシステムプロンプトを組み立てる。gate_notice はその日の買いの止めの知らせ。"""
    return f"""あなたは株式トレードの判断を行う AI エージェントです。

## 役割
tool を使って当日の予測・ポジション・残高・リスク状態を収集し、
buy/sell/hold の判断を行い、発注を実行してください。

## 制約（厳守）
- is_allowed=false のときは新規の buy をしない（保有の sell は可）
- 1セッションの最大発注数: {MAX_ORDERS_PER_RUN} 件
- 既に保有中の銘柄に追加買いしない
- 確信のない銘柄は hold（見送り）を選択する
- place_order の reasoning には判断根拠を必ず記述する

## 推奨フロー
1. get_risk_status で発注可否を確認
2. get_predictions でシグナル候補を取得
3. get_balance / get_positions で資金状況を把握
4. 各候補を評価して place_order を呼ぶ（または見送る）
5. 全判断が終わったらセッションを終了する
{gate_notice}
market: {market}, mode: {mode}
"""


TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_predictions",
        "description": (
            "当日の ML 予測結果から閾値通過候補を取得する。"
            "buy 候補（multi_horizon_score が effective_buy_threshold 以上）と"
            "sell 候補（保有中かつ effective_sell_threshold 以下）を返す。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "description": "対象マーケット。例: 'jp', 'us'",
                }
            },
            "required": ["market"],
        },
    },
    {
        "name": "get_positions",
        "description": "現在の保有ポジション一覧を返す。symbol, qty, avg_price, current_price を含む。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_balance",
        "description": "現在の現金余力（円）を返す。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_risk_status",
        "description": (
            "リスクゲートの評価結果を返す。"
            "is_allowed=false の場合は当日の新規の buy が禁止されている（保有の sell は可）。"
            "daily_loss, consecutive_losses, position_count なども含む。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "place_order",
        "description": (
            "指定銘柄の注文を発注する。"
            "RiskManager のゲートチェックとポジションサイジングは Python 側で自動適用される。"
            "ゲート不合格の日は buy だけが止まり、保有の sell は受け付ける。"
            "side は 'buy' または 'sell' を指定する。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "銘柄コード。例: '7203', 'AAPL'",
                },
                "side": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "発注方向",
                },
                "reasoning": {
                    "type": "string",
                    "description": "この発注判断の理由（ログ保存用）",
                },
            },
            "required": ["symbol", "side", "reasoning"],
        },
    },
]
