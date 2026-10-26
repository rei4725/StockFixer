"""
ClaudeTrader に渡す tool のスキーマ定義

claude_agent.py から分離（ファイル行数ゲート対応）。
"""

from __future__ import annotations

from typing import Any

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
