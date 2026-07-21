"""
ClaudeTrader — Claude Opus を用いた LLM ベーストレード判断エージェント

ML モデルの予測結果（閾値通過候補）を入力として、Claude が tool use で
ポジション・残高・リスク状態を収集し、推論付きで buy/sell/hold を判断する。

設計上の制約:
- RiskManager ゲートは place_order tool 内で Python 側から強制通過させる
  （Claude が意図的にバイパスすることはできない）
- CLAUDE_TRADER_ENABLED=False がデフォルト。明示的に有効化するまで動作しない
- paper mode 専用を推奨。live 使用は自己責任
- extended thinking で推論ログを claude_reasoning テーブルに保存（監査目的）
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd

from config.settings import (
    CLAUDE_TRADER_MAX_TOKENS,
    CLAUDE_TRADER_MODEL,
    CLAUDE_TRADER_THINKING_BUDGET,
    MAX_ORDERS_PER_RUN,
    MIN_CHANGE_RATIO,
)
from src.trading.brokers.base import BrokerBase, BrokerError, OrderSide
from src.trading.execution import (
    _attach_dynamic_thresholds,
    _choose_order_params,
    _get_held_symbols,
    _load_latest_predictions,
    _record_order,
    _resolve_kelly_params,
)
from src.trading.risk_manager import RiskManager
from src.trading.signal_generator import apply_multi_horizon_score_column
from src.trading.types import TradingGateStatus
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# tool use の最大ターン数（無限ループ防止）
_MAX_AGENT_TURNS = 20

# Claude に渡す予測候補の上限行数（トークン節約）
_MAX_PREDICTION_ROWS = 50


# ---------------------------------------------------------------------------
# Tool スキーマ定義
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
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
            "is_allowed=false の場合は当日の発注が禁止されている。"
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


# ---------------------------------------------------------------------------
# Tool ハンドラー
# ---------------------------------------------------------------------------


def _handle_get_predictions(market: str, broker: BrokerBase) -> dict[str, Any]:
    """閾値通過候補の予測を返す。トークン節約のため上位 N 行に絞る。"""
    try:
        predictions = _load_latest_predictions(market)
        if predictions.empty:
            return {"candidates": [], "message": "予測結果なし"}

        predictions = _attach_dynamic_thresholds(predictions)
        predictions = apply_multi_horizon_score_column(predictions)

        held_symbols = _get_held_symbols(broker)

        buy_candidates = predictions[
            (~predictions["symbol"].isin(held_symbols))
            & (predictions["multi_horizon_score"] >= predictions["effective_buy_threshold"])
        ].head(_MAX_PREDICTION_ROWS)

        sell_candidates = predictions[
            (predictions["symbol"].isin(held_symbols))
            & (predictions["multi_horizon_score"] <= predictions["effective_sell_threshold"])
        ]

        def _to_records(df: pd.DataFrame) -> list[dict]:
            cols = [
                "market",
                "symbol",
                "current_price",
                "diff_ratio",
                "diff_ratio_3d",
                "diff_ratio_5d",
                "diff_ratio_10d",
                "confluence_score",
                "confidence_ratio",
                "multi_horizon_score",
                "effective_buy_threshold",
                "effective_sell_threshold",
            ]
            available = [c for c in cols if c in df.columns]
            return df[available].to_dict(orient="records")

        return {
            "buy_candidates": _to_records(buy_candidates),
            "sell_candidates": _to_records(sell_candidates),
            "held_symbols": sorted(held_symbols),
            "threshold_scale": (
                float(predictions["threshold_scale"].iloc[0]) if not predictions.empty else 1.0
            ),
        }
    except Exception as e:
        logger.error("[claude_agent] get_predictions エラー: %s", e, exc_info=True)
        return {"error": str(e)}


def _handle_get_positions(broker: BrokerBase) -> dict[str, Any]:
    try:
        return {"positions": broker.get_positions()}
    except Exception as e:
        logger.error("[claude_agent] get_positions エラー: %s", e, exc_info=True)
        return {"error": str(e)}


def _handle_get_balance(broker: BrokerBase) -> dict[str, Any]:
    try:
        return {"balance": broker.get_balance()}
    except Exception as e:
        logger.error("[claude_agent] get_balance エラー: %s", e, exc_info=True)
        return {"error": str(e)}


def _handle_get_risk_status(risk: RiskManager) -> dict[str, Any]:
    try:
        gate: TradingGateStatus = risk.evaluate_trading_gate()
        return {
            "is_allowed": gate.is_allowed,
            "stop_active": gate.stop_active,
            "reason_code": gate.reason_code,
            "reason": gate.reason,
            "daily_loss": gate.daily_loss,
            "daily_loss_limit": gate.daily_loss_limit,
            "consecutive_losses": gate.consecutive_losses,
            "position_count": gate.position_count,
            "max_positions": gate.max_positions,
        }
    except Exception as e:
        logger.error("[claude_agent] get_risk_status エラー: %s", e, exc_info=True)
        return {"error": str(e)}


def _handle_place_order(
    symbol: str,
    side_str: str,
    reasoning: str,
    market: str,
    broker: BrokerBase,
    risk: RiskManager,
    predictions_cache: pd.DataFrame,
    mode: str,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """
    RiskManager ゲートとポジションサイジングを強制適用して注文を発注する。
    Claude がこれらをバイパスする手段はない。
    """
    # 発注方向の変換
    side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

    # RiskManager 再チェック（Claude の判断時点と現在の間にレースコンディションが起きても安全）
    gate = risk.evaluate_trading_gate()
    if not gate.is_allowed:
        msg = f"RiskManager ゲート不合格のためスキップ: {gate.reason}"
        logger.warning("[claude_agent] %s → %s", symbol, msg)
        stats["skipped"] += 1
        return {"status": "skipped", "reason": msg}

    # 予測行を取得
    row = None
    if not predictions_cache.empty and "symbol" in predictions_cache.columns:
        matches = predictions_cache[predictions_cache["symbol"] == symbol]
        if not matches.empty:
            row = matches.iloc[0]

    if row is None:
        stats["skipped"] += 1
        return {"status": "skipped", "reason": f"{symbol} の予測データが見つかりません"}

    current_price = float(row.get("current_price") or 0.0)
    if current_price <= 0:
        stats["skipped"] += 1
        return {"status": "skipped", "reason": f"{symbol} の現在価格が取得できません"}

    diff_ratio = float(row.get("diff_ratio") or 0.0)
    if abs(diff_ratio) < MIN_CHANGE_RATIO:
        stats["skipped_min_change"] = stats.get("skipped_min_change", 0) + 1
        stats["skipped"] += 1
        return {
            "status": "skipped",
            "reason": f"diff_ratio={diff_ratio:.3%} < min_change={MIN_CHANGE_RATIO:.3%}",
        }

    if side == OrderSide.BUY:
        held = _get_held_symbols(broker)
        if symbol in held:
            stats["skipped"] += 1
            return {"status": "skipped", "reason": f"{symbol} は既に保有中"}

        kelly_win_rate, kelly_avg_win, kelly_avg_loss = _resolve_kelly_params(
            str(row["market"]), symbol
        )
        confidence = float(row.get("confidence_ratio") or 1.0)
        qty = risk.calc_position_size(
            symbol,
            current_price,
            win_rate=kelly_win_rate,
            avg_win=kelly_avg_win,
            avg_loss=kelly_avg_loss,
            confidence_ratio=confidence,
        )
        if qty <= 0:
            stats["skipped"] += 1
            return {
                "status": "skipped",
                "reason": f"{symbol}: ポジションサイズ 0（残高不足or上限）",
            }
    else:
        positions = broker.get_positions()
        pos = next((p for p in positions if p["symbol"].replace(".T", "") == symbol), None)
        if pos is None or pos.get("qty", 0) <= 0:
            stats["skipped"] += 1
            return {"status": "skipped", "reason": f"{symbol} の保有ポジションなし"}
        qty = pos["qty"]

    try:
        order_type, order_price, order_reason, order_session = _choose_order_params(
            market=str(row["market"]),
            symbol=symbol,
            side=side,
            current_price=current_price,
        )
        result = broker.send_order(symbol, side, qty, price=order_price, order_type=order_type)
        _record_order(
            market=str(row["market"]),
            predicted_at=str(row.get("predicted_at", "")),
            symbol=symbol,
            side=side,
            qty=qty,
            signal_price=current_price,
            order_price=order_price,
            order_type=order_type,
            order_result=result,
            broker=broker,
            mode=mode,
            order_session=order_session,
        )
        if side == OrderSide.BUY:
            stats["buy_orders"] += 1
        else:
            stats["sell_orders"] += 1
        stats["total_turnover"] = stats.get("total_turnover", 0.0) + current_price * qty

        logger.info(
            "[claude_agent] %s %s %d株 @ %s(%s) | 理由: %s",
            side.name,
            symbol,
            qty,
            order_type.name,
            order_reason,
            reasoning,
        )
        return {
            "status": "placed",
            "symbol": symbol,
            "side": side_str,
            "qty": qty,
            "order_type": order_type.name,
            "order_price": order_price,
            "order_id": result.get("order_id", ""),
        }
    except BrokerError as e:
        logger.error("[claude_agent] 発注エラー %s: %s", symbol, e, exc_info=True)
        stats["errors"] += 1
        return {"status": "error", "reason": str(e)}


# ---------------------------------------------------------------------------
# 推論ログ保存
# ---------------------------------------------------------------------------


def _save_reasoning_log(run_id: str, market: str, thinking_text: str, summary: str) -> None:
    """Extended thinking の推論ログを claude_reasoning テーブルに保存する。"""  # noqa: D401
    try:
        with _db_connection() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS claude_reasoning (
                    run_id      VARCHAR PRIMARY KEY,
                    market      VARCHAR,
                    thinking    TEXT,
                    summary     TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
            con.execute(
                """
                INSERT OR REPLACE INTO claude_reasoning (run_id, market, thinking, summary)
                VALUES (?, ?, ?, ?)
                """,
                [run_id, market, thinking_text, summary],
            )
    except Exception:
        logger.warning("[claude_agent] 推論ログ保存に失敗しました", exc_info=True)


# ---------------------------------------------------------------------------
# メインエントリーポイント
# ---------------------------------------------------------------------------


def run_claude_trader(
    broker: BrokerBase,
    market: str = "jp",
    mode: str = "paper",
) -> dict[str, Any]:
    """
    Claude Opus を用いたトレード判断エージェントを実行する。

    Args:
        broker: BrokerBase 実装（推奨: PaperBroker）
        market: 対象マーケット
        mode: "paper" or "live"

    Returns:
        buy_orders, sell_orders, skipped, errors 等の実行統計
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic パッケージが必要です。`pip install anthropic>=0.50.0` を実行してください。"
        )

    run_id = str(uuid.uuid4())[:12]
    logger.info(
        "=== Claude トレーダー開始: run_id=%s market=%s mode=%s model=%s ===",
        run_id,
        market,
        mode,
        CLAUDE_TRADER_MODEL,
    )

    stats: dict[str, Any] = {
        "buy_orders": 0,
        "sell_orders": 0,
        "skipped": 0,
        "skipped_min_change": 0,
        "errors": 0,
        "trading_stopped": False,
        "stop_reason": None,
        "total_turnover": 0.0,
        "run_id": run_id,
    }

    # トークン取得
    try:
        broker.get_token()
    except BrokerError as e:
        logger.error("[claude_agent] トークン取得失敗: %s", e, exc_info=True)
        return stats

    risk = RiskManager(broker, market=market)
    risk.update_peak_balance()

    # 発注可否チェック（早期リターン）
    gate = risk.evaluate_trading_gate()
    if not gate.is_allowed:
        logger.warning("[claude_agent] リスクゲート不合格 → スキップ: %s", gate.reason)
        stats.update({"trading_stopped": gate.stop_active, "stop_reason": gate.reason})
        return stats

    # 予測キャッシュ（place_order ハンドラーで参照）
    predictions_cache = _load_latest_predictions(market)
    if not predictions_cache.empty:
        predictions_cache = _attach_dynamic_thresholds(predictions_cache)
        predictions_cache = apply_multi_horizon_score_column(predictions_cache)

    # システムプロンプト
    system_prompt = f"""あなたは株式トレードの判断を行う AI エージェントです。

## 役割
tool を使って当日の予測・ポジション・残高・リスク状態を収集し、
buy/sell/hold の判断を行い、発注を実行してください。

## 制約（厳守）
- is_allowed=false のときは発注しない
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

market: {market}, mode: {mode}
"""

    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "当日のトレード判断を開始してください。"},
    ]

    thinking_log: list[str] = []
    turn = 0

    while turn < _MAX_AGENT_TURNS:
        turn += 1

        thinking_cfg: dict[str, Any] = (
            {"type": "enabled", "budget_tokens": CLAUDE_TRADER_THINKING_BUDGET}
            if CLAUDE_TRADER_THINKING_BUDGET > 0
            else {"type": "disabled"}
        )

        response = client.messages.create(
            model=CLAUDE_TRADER_MODEL,
            max_tokens=CLAUDE_TRADER_MAX_TOKENS,
            system=system_prompt,
            thinking=thinking_cfg,
            tools=_TOOLS,
            messages=messages,
        )

        # thinking ブロックをログ収集
        for block in response.content:
            if block.type == "thinking":
                thinking_log.append(block.thinking)

        if response.stop_reason == "end_turn":
            logger.info("[claude_agent] エージェント終了 (turn=%d)", turn)
            break

        if response.stop_reason != "tool_use":
            logger.warning("[claude_agent] 予期しない stop_reason=%s", response.stop_reason)
            break

        # tool 呼び出し処理
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            logger.debug("[claude_agent] tool_use: %s input=%s", tool_name, tool_input)

            if tool_name == "get_predictions":
                result = _handle_get_predictions(tool_input.get("market", market), broker)

            elif tool_name == "get_positions":
                result = _handle_get_positions(broker)

            elif tool_name == "get_balance":
                result = _handle_get_balance(broker)

            elif tool_name == "get_risk_status":
                result = _handle_get_risk_status(risk)

            elif tool_name == "place_order":
                result = _handle_place_order(
                    symbol=str(tool_input.get("symbol", "")),
                    side_str=str(tool_input.get("side", "buy")),
                    reasoning=str(tool_input.get("reasoning", "")),
                    market=market,
                    broker=broker,
                    risk=risk,
                    predictions_cache=predictions_cache,
                    mode=mode,
                    stats=stats,
                )
            else:
                result = {"error": f"未知の tool: {tool_name}"}

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    else:
        logger.warning("[claude_agent] 最大ターン数 %d に到達しました", _MAX_AGENT_TURNS)

    # 推論ログを保存
    if thinking_log:
        _save_reasoning_log(
            run_id=run_id,
            market=market,
            thinking_text="\n\n---\n\n".join(thinking_log),
            summary=(
                f"buy={stats['buy_orders']} sell={stats['sell_orders']} "
                f"skipped={stats['skipped']} errors={stats['errors']}"
            ),
        )

    logger.info(
        "=== Claude トレーダー完了: 買い=%d 売り=%d スキップ=%d エラー=%d ===",
        stats["buy_orders"],
        stats["sell_orders"],
        stats["skipped"],
        stats["errors"],
    )
    return stats
