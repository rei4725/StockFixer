"""紙に注文を出す道（自動発注・期限切れの手じまい・ルール売買）が使う Broker を選ぶ

選び方はポートとして差し込む（#1029）。既定は PaperBroker だけを返し、設定の値を読まない。
本番は orchestration の wire_ports() が make_configured_order_broker を差し込み、
ACCOUNT_SHADOW などの値を見て ShadowBroker を選ぶ。テストは差し込まないので、
OS の環境変数に値があっても口座アプリの gateway には届かない。
"""

from collections.abc import Callable

import config.settings as cfg
from src.domain.ports import MarketDataPort, NotificationPort, TradeDiffSink
from src.trading.brokers.account.gateway import AccountGatewayBroker
from src.trading.brokers.account.shadow import ShadowBroker
from src.trading.brokers.base import BrokerBase
from src.trading.brokers.paper import paper_broker
from src.utils.logger import get_logger

logger = get_logger(__name__)

OrderBrokerSelector = Callable[..., BrokerBase]


def make_paper_order_broker(
    *,
    job: str,
    market_data_port: MarketDataPort,
    trade_diff_sink: TradeDiffSink,
    notifier: NotificationPort | None = None,
) -> BrokerBase:
    """既定の選び。設定の値によらず PaperBroker を返す"""
    # 呼ぶ時にモジュールから引く（既存のテストが paper_broker.PaperBroker を差し替える）
    return paper_broker.PaperBroker(
        market_data_port=market_data_port, trade_diff_sink=trade_diff_sink
    )


def make_configured_order_broker(
    *,
    job: str,
    market_data_port: MarketDataPort,
    trade_diff_sink: TradeDiffSink,
    notifier: NotificationPort | None = None,
) -> BrokerBase:
    """ACCOUNT_SHADOW が true で URL とトークンがあれば ShadowBroker、無ければ今までどおり PaperBroker"""
    paper = make_paper_order_broker(
        job=job, market_data_port=market_data_port, trade_diff_sink=trade_diff_sink
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


_selector: OrderBrokerSelector = make_paper_order_broker


def set_order_broker_selector(selector: OrderBrokerSelector) -> None:
    """注文の Broker の選びを差し込む（本番は wire_ports()、テストは conftest）"""
    global _selector
    _selector = selector


def make_order_broker(
    *,
    job: str,
    market_data_port: MarketDataPort,
    trade_diff_sink: TradeDiffSink,
    notifier: NotificationPort | None = None,
) -> BrokerBase:
    """差し込まれた選びで Broker を作る。差し込みが無ければ PaperBroker"""
    return _selector(
        job=job,
        market_data_port=market_data_port,
        trade_diff_sink=trade_diff_sink,
        notifier=notifier,
    )
