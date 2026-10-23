"""紙に注文を出す道（自動発注・期限切れの手じまい・ルール売買）が使う Broker を選ぶ"""

import config.settings as cfg
from src.domain.ports import MarketDataPort, NotificationPort, TradeDiffSink
from src.trading.brokers.account.gateway import AccountGatewayBroker
from src.trading.brokers.account.shadow import ShadowBroker
from src.trading.brokers.base import BrokerBase
from src.trading.brokers.paper import paper_broker
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_order_broker(
    *,
    job: str,
    market_data_port: MarketDataPort,
    trade_diff_sink: TradeDiffSink,
    notifier: NotificationPort | None = None,
) -> BrokerBase:
    """ACCOUNT_SHADOW が true で URL とトークンがあれば ShadowBroker、無ければ今までどおり PaperBroker"""
    # 呼ぶ時にモジュールから引く（既存のテストが paper_broker.PaperBroker を差し替える）
    paper = paper_broker.PaperBroker(
        market_data_port=market_data_port, trade_diff_sink=trade_diff_sink
    )
    if not cfg.ACCOUNT_SHADOW:
        return paper
    if not (cfg.ACCOUNT_GATEWAY_URL and cfg.ACCOUNT_GATEWAY_TOKEN):
        logger.warning(
            "[shadow] ACCOUNT_SHADOW=true だが ACCOUNT_GATEWAY_URL か ACCOUNT_GATEWAY_TOKEN が無い。"
            "紙だけで動く"
        )
        return paper
    gateway = AccountGatewayBroker(
        cfg.ACCOUNT_GATEWAY_URL,
        cfg.ACCOUNT_GATEWAY_TOKEN,
        job=job,
        timeout=cfg.ACCOUNT_GATEWAY_TIMEOUT_SECONDS,
        notifier=notifier,
    )
    return ShadowBroker(paper, gateway, job=job)
