import os

import discord
from discord.ext import commands
from discord.utils import escape_markdown
from dotenv import load_dotenv

from src.orchestration.types import SchedulerJobStatus
from src.prediction.types import PredictionResult
from src.reporting.discord.discord_formatters import convert_df_for_discord
from src.reporting.discord.discord_text import DISCORD_TEXT_LIMIT, split_text_chunks
from src.reporting.query_service import (
    get_latest_market_prediction_snapshots,
    get_monthly_report_summary,
    get_ranked_prediction_results,
    get_scheduler_job_statuses,
    get_signal_snapshot,
    get_watchlist_prediction_view,
)
from src.reporting.types import (
    MonthlyReportSummary,
    WatchlistPredictionRow,
    WatchlistPredictionView,
)
from src.utils.japan_time import format_jst_from_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)

load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
INTENTS = discord.Intents.default()
INTENTS.message_content = True

bot = commands.Bot(command_prefix="!", intents=INTENTS)

TOP10_LABEL = "差異割合上位10銘柄"
WORST10_LABEL = "差異割合ワースト10銘柄"
WATCHLIST_LABEL = "監視対象銘柄"


async def send_chunked_channel_message(channel, text: str, *, preserve_lines: bool = False) -> None:
    for chunk in split_text_chunks(text, limit=DISCORD_TEXT_LIMIT, preserve_lines=preserve_lines):
        await channel.send(chunk)


def build_code_block_message(title: str, body: str) -> str:
    return f"=== {title} ===\n```text\n{body}\n```"


def build_ranked_stock_message(market: str, label: str, table_text: str) -> str:
    return build_code_block_message(f"{market} {label}", table_text)


def build_watchlist_message(table_text: str) -> str:
    return build_code_block_message(WATCHLIST_LABEL, table_text)


def determine_signal_label(diff_ratio: float) -> str:
    if diff_ratio > 0.005:
        return "⬆️ Buy"
    if diff_ratio < -0.005:
        return "⬇️ Sell"
    return "⏺️ Hold"


def build_signal_lines(
    market: str,
    symbol: str,
    current_price: float,
    predicted_price: float,
    diff_ratio: float,
    model_count: int,
    shap_result: dict | None = None,
) -> list[str]:
    import math

    current_disp = math.floor(current_price * 1000) / 1000
    pred_disp = math.floor(predicted_price * 1000) / 1000
    change_pct = f"{diff_ratio * 100:.2g}%"
    signal = determine_signal_label(diff_ratio)

    lines = [
        f"=== {market.upper()} / {symbol} ===",
        f"現在値  : {current_disp}",
        f"予想終値: {pred_disp}",
        f"予想変化率: {change_pct}",
        f"シグナル : {signal}",
        f"使用モデル: {model_count}モデルの平均",
    ]
    if shap_result:
        lines.append("")
        lines.append(f"[SHAP 寄与度 Top5 - 方向:{shap_result['direction']}]")
        for feat in shap_result["top_features"]:
            arrow = "▲" if feat["shap_value"] > 0 else "▼"
            lines.append(f"  {arrow} {feat['feature'][:30]:<30}  {feat['shap_value']:+.5f}")
    return lines


def build_scheduler_status_lines(statuses: list[SchedulerJobStatus]) -> list[str]:
    lines = ["=== スケジューラ状態 ==="]
    for job_status in statuses:
        timestamp = (
            format_jst_from_iso(job_status.last_run_at)
            if job_status.last_run_at
            else "実行履歴なし"
        )
        lines.append(job_status.label)
        lines.append(f"  最終実行: {timestamp}  [状態: {job_status.status}]")
    return lines


def build_prediction_table_text(results: list[PredictionResult]) -> str:
    if not results:
        return ""
    return convert_df_for_discord(PredictionResult.to_dataframe(results)).to_string(index=False)


def get_top10_diff_stocks_message(market: str, rank_type: str, predicted_at: str = None) -> str:
    """DBから予測結果を取得してDiscord表示用テキストに変換する"""
    return build_prediction_table_text(
        get_ranked_prediction_results(market, rank_type, predicted_at)
    )


async def handle_forecast_command(message):
    latest_ts, snapshots = get_latest_market_prediction_snapshots()
    if latest_ts is None:
        await message.channel.send(
            escape_markdown("予測結果が見つかりませんでした。"), allowed_mentions=None
        )
        return

    if not snapshots:
        await message.channel.send(
            escape_markdown("予測結果が見つかりませんでした。"), allowed_mentions=None
        )
        return

    for snapshot in snapshots:
        table_text = build_prediction_table_text(snapshot.top_results)
        if not table_text:
            continue
        await send_chunked_channel_message(
            message.channel,
            build_ranked_stock_message(snapshot.market, TOP10_LABEL, table_text),
        )

    for snapshot in snapshots:
        table_text = build_prediction_table_text(snapshot.worst_results)
        if not table_text:
            continue
        await send_chunked_channel_message(
            message.channel,
            build_ranked_stock_message(snapshot.market, WORST10_LABEL, table_text),
        )


@bot.event
async def on_ready():
    logger.info("Bot起動完了: %s", bot.user)


def build_watchlist_prediction_text(view: WatchlistPredictionView) -> str:
    if not view.is_success:
        return view.error_message or "[エラー] 監視対象予測処理に失敗しました"
    df = convert_df_for_discord(WatchlistPredictionRow.to_dataframe(view.rows))
    return df.to_string(index=False)


async def handle_watchnext_command(message):
    text = build_watchlist_prediction_text(get_watchlist_prediction_view())
    await send_chunked_channel_message(message.channel, build_watchlist_message(text))


async def handle_signal_command(message, parts: list):
    """
    /signal <symbol> [market] [--explain]
    指定銘柄の最新予測シグナルを表示する。
    --explain を付けると SHAP による特徴量寄与度トップ5を追加表示する。
    """
    if len(parts) < 2:
        await message.channel.send(
            escape_markdown(
                "使い方: /signal <銘柄コード> [market] [--explain]  例: /signal AAPL us --explain"
            ),
            allowed_mentions=None,
        )
        return

    symbol = parts[1].upper()
    market = parts[2].lower() if len(parts) >= 3 and not parts[2].startswith("--") else "us"
    explain = "--explain" in parts

    await message.channel.send(
        escape_markdown(f"計算中... {market}/{symbol}"), allowed_mentions=None
    )
    snapshot = get_signal_snapshot(market, symbol, explain=explain)
    if snapshot is None:
        await message.channel.send(
            escape_markdown(f"モデルが見つかりませんでした: {market}/{symbol}"),
            allowed_mentions=None,
        )
        return

    prediction = snapshot.prediction
    lines = build_signal_lines(
        prediction.market,
        prediction.symbol,
        prediction.current_price,
        prediction.avg_pred_price,
        prediction.diff_ratio,
        prediction.model_count,
        shap_result=(
            {
                "direction": snapshot.shap_direction,
                "top_features": [
                    {"feature": feature.feature, "shap_value": feature.shap_value}
                    for feature in snapshot.top_features
                ],
            }
            if snapshot.shap_direction is not None and snapshot.top_features
            else None
        ),
    )
    if explain and not snapshot.top_features:
        lines.append("(SHAP 計算スキップ: モデルまたはライブラリが未対応)")

    msg = "```text\n" + "\n".join(lines) + "\n```"
    await message.channel.send(msg)


def build_monthly_report_lines(summary: MonthlyReportSummary) -> list[str]:
    def _fmt_pct(val: float | None) -> str:
        return f"{val * 100:.2f}%" if val is not None else "データなし"

    def _fmt_f2(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "データなし"

    target_kpi = (
        "達成" if (summary.sharpe_ratio is not None and summary.sharpe_ratio >= 1.0) else "未達"
    )
    mdd_kpi = (
        "達成"
        if (summary.max_drawdown is not None and abs(summary.max_drawdown) <= 0.15)
        else "未達"
    )

    lines = [
        f"=== 月次KPIレポート {summary.target_month} ===",
        "",
        "【最重要KPI】",
        f"  Net Return   : {_fmt_pct(summary.net_return)}",
        f"  Max Drawdown : {_fmt_pct(summary.max_drawdown)}  (目標: 15%以下  → {mdd_kpi})",
        f"  Sharpe Ratio : {_fmt_f2(summary.sharpe_ratio)}  (目標: 1.0以上  → {target_kpi})",
        "",
        "【補助KPI】",
        f"  Hit Rate     : {_fmt_pct(summary.hit_rate)}  (直近30日)",
        f"  Avg Slippage : {_fmt_pct(summary.avg_slippage)}  (直近30日)",
        "",
        "【メタ情報】",
        f"  対象銘柄数   : {summary.symbol_count or 'N/A'}",
        f"  WFスナップ   : {summary.wf_snapshot_file or 'N/A'}",
        f"  生成日時     : {summary.generated_at[:19]}",
    ]
    return lines


async def handle_monthlyreport_command(message):
    """
    /monthlyreport [YYYY-MM]
    月次KPIサマリーを表示する。
    """
    parts = message.content.split()
    target_month = parts[1] if len(parts) >= 2 else None

    await message.channel.send(
        escape_markdown("月次レポートを集計中..."),
        allowed_mentions=None,
    )
    try:
        summary = get_monthly_report_summary(target_month)
    except Exception as e:
        await message.channel.send(
            escape_markdown(f"月次レポートの取得に失敗しました: {e}"),
            allowed_mentions=None,
        )
        return

    msg = "```text\n" + "\n".join(build_monthly_report_lines(summary)) + "\n```"
    await send_chunked_channel_message(message.channel, msg)


async def handle_status_command(message):
    """
    /status
    スケジューラの最終実行時刻を表示する。
    """
    from src.utils.data_path_utils import get_results_dir

    state_file = os.path.join(get_results_dir(), "scheduler_queue_state.json")
    if not os.path.exists(state_file):
        await message.channel.send(
            escape_markdown(
                "スケジューラ状態ファイルが見つかりません。スケジューラが起動されていない可能性があります。"
            ),
            allowed_mentions=None,
        )
        return

    try:
        statuses = get_scheduler_job_statuses(state_file)
    except Exception as e:
        await message.channel.send(
            escape_markdown(f"状態ファイルの読み込みに失敗しました: {e}"),
            allowed_mentions=None,
        )
        return

    msg = "```text\n" + "\n".join(build_scheduler_status_lines(statuses)) + "\n```"
    await message.channel.send(msg)


_DRIFT_THRESHOLD_USAGE = (
    "使い方:\n"
    "  /drift_threshold show            — 現在の閾値を表示\n"
    "  /drift_threshold set mae <値>    — MAE 閾値を更新 (例: 0.03)\n"
    "  /drift_threshold set hit_rate <値> — Hit Rate 閾値を更新 (例: 0.40)"
)


async def handle_drift_threshold_command(message, parts: list[str]) -> None:
    """
    /drift_threshold show
    /drift_threshold set mae <value>
    /drift_threshold set hit_rate <value>
    """
    import os

    from src.utils.db.system_config import get_config_value, set_config_value

    subcommand = parts[1] if len(parts) > 1 else ""

    if subcommand == "show":
        mae_default = os.environ.get("DRIFT_MAE_THRESHOLD", "0.02")
        hr_default = os.environ.get("DRIFT_HIT_RATE_THRESHOLD", "0.45")
        mae = get_config_value("drift.mae_threshold", mae_default)
        hit_rate = get_config_value("drift.hit_rate_threshold", hr_default)
        lines = [
            "=== ドリフト監視閾値 ===",
            f"MAE 閾値      : {mae}",
            f"Hit Rate 閾値 : {hit_rate}",
        ]
        await message.channel.send("```text\n" + "\n".join(lines) + "\n```")
        return

    if subcommand == "set" and len(parts) >= 4:
        kind = parts[2]
        raw = parts[3]
        try:
            val = float(raw)
        except ValueError:
            await message.channel.send(
                escape_markdown(f"値が数値ではありません: {raw}"), allowed_mentions=None
            )
            return
        if val <= 0 or val >= 1:
            await message.channel.send(
                escape_markdown("値は 0 より大きく 1 未満にしてください。"), allowed_mentions=None
            )
            return
        if kind == "mae":
            set_config_value("drift.mae_threshold", str(val))
            await message.channel.send(
                escape_markdown(f"MAE 閾値を {val} に更新しました。"), allowed_mentions=None
            )
        elif kind == "hit_rate":
            set_config_value("drift.hit_rate_threshold", str(val))
            await message.channel.send(
                escape_markdown(f"Hit Rate 閾値を {val} に更新しました。"), allowed_mentions=None
            )
        else:
            await message.channel.send(
                escape_markdown(f"不明なキー: {kind}\n" + _DRIFT_THRESHOLD_USAGE),
                allowed_mentions=None,
            )
        return

    await message.channel.send(escape_markdown(_DRIFT_THRESHOLD_USAGE), allowed_mentions=None)


@bot.event
async def on_message(message):
    try:
        if message.author.bot:
            return

        parts = message.content.split()
        command = parts[0] if parts else ""

        if command == "/forecast":
            await handle_forecast_command(message)
        elif command == "/WatchNext":
            await handle_watchnext_command(message)
        elif command == "/signal":
            await handle_signal_command(message, parts)
        elif command == "/status":
            await handle_status_command(message)
        elif command == "/monthlyreport":
            await handle_monthlyreport_command(message)
        elif command == "/drift_threshold":
            await handle_drift_threshold_command(message, parts)
        else:
            await message.channel.send(
                escape_markdown(f"受信: {message.content}"), allowed_mentions=None
            )

        await bot.process_commands(message)
    except Exception:
        await message.channel.send(escape_markdown("エラーが発生しました"), allowed_mentions=None)
        logger.error("Error in on_message", exc_info=True)


if __name__ == "__main__":
    if not TOKEN:
        logger.critical("DISCORD_BOT_TOKEN環境変数が設定されていません。")
    else:
        bot.run(TOKEN)
