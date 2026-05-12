"""
StockFixer カスタム例外階層

各層が意味のある例外型で失敗を表現できるよう、共通基底クラスとサブクラスを定義する。
"""


class StockFixerError(Exception):
    """StockFixer 全層共通の基底例外"""


class DataFetchError(StockFixerError):
    """data 層・yfinance 障害"""


class ModelError(StockFixerError):
    """models 層・学習/予測失敗"""


class BrokerError(StockFixerError):
    """brokers 層・証券 API 障害"""


class PipelineError(StockFixerError):
    """services 層・パイプライン失敗"""


class RiskError(StockFixerError):
    """リスクチェックによる取引拒否"""


class ConfigError(StockFixerError):
    """optimal_params.json 等の設定バリデーション失敗"""
