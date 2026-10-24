"""段3の影の運用の日報ジョブ（毎営業日 17:00。16:30 のルール売買の後）"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_daily_account_shadow_report() -> None:
    """紙と口座アプリのその日の注文・持ち高・評価額を比べ、results/account_shadow/ に書く"""
    import config.settings as cfg

    if not cfg.ACCOUNT_SHADOW:
        logger.info("[shadow] ACCOUNT_SHADOW が false のため日報を書かない")
        return
    if not (cfg.ACCOUNT_GATEWAY_URL and cfg.ACCOUNT_GATEWAY_TOKEN):
        logger.warning(
            "[shadow] ACCOUNT_GATEWAY_URL か ACCOUNT_GATEWAY_TOKEN が無いため日報を書かない"
        )
        return

    # 差し替えやすいよう、モジュールから引く（daily.py と同じく関数の中で import する）
    from src.infrastructure import yfinance_market_data_adapter
    from src.infrastructure.persistence import trade_diff_repository
    from src.trading.brokers.account import gateway, shadow_report
    from src.trading.brokers.paper import paper_broker
    from src.utils.japan_time import now_jst

    day = now_jst().date()
    paper = paper_broker.PaperBroker(
        market_data_port=yfinance_market_data_adapter.YFinanceMarketDataAdapter(),
        trade_diff_sink=trade_diff_repository.PostgresTradeDiffSink(),
    )
    # 日報は読むだけ。読めずに止めの印が立っても、17:00 の後に注文は無い
    account = gateway.AccountGatewayBroker(
        cfg.ACCOUNT_GATEWAY_URL,
        cfg.ACCOUNT_GATEWAY_TOKEN,
        job="report",
        timeout=cfg.ACCOUNT_GATEWAY_TIMEOUT_SECONDS,
    )
    report = shadow_report.build_shadow_report(
        day, shadow_report.read_shadow_log(day), paper, account
    )
    path = shadow_report.write_shadow_report(day, report)
    logger.info("[shadow] 日報を書いた: %s summary=%s", path, report.get("summary"))
