"""
ファイルパス制御モジュール

アプリ全体のファイルパス設定を一元管理する。
他のモジュールはこのファイルの関数を使用してパスを取得すること。
"""
import os
from datetime import datetime

# ===== ベースディレクトリ =====
# pythonフォルダの絶対パス（このファイルから2階層上）
_PYTHON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_python_root() -> str:
    """pythonフォルダのルートパスを返す"""
    return _PYTHON_ROOT


# ===== データ関連パス =====
def get_data_dir() -> str:
    """データ保存用ディレクトリの絶対パスを返す"""
    return os.path.join(_PYTHON_ROOT, "data")


def get_data_subdir(market: str, symbol: str) -> str:
    """指定market,symbolのデータサブディレクトリの絶対パスを返す"""
    return os.path.join(get_data_dir(), f"{market}_{symbol}")


# ===== モデル関連パス =====
def get_models_dir() -> str:
    """モデル保存用ディレクトリの絶対パスを返す"""
    return os.path.join(_PYTHON_ROOT, "models")


def get_models_subdir(market: str, symbol: str) -> str:
    """指定market,symbolのモデルサブディレクトリの絶対パスを返す"""
    return os.path.join(get_models_dir(), f"{market}_{symbol}")


def get_model_path(market: str, symbol: str, model_name: str) -> str:
    """指定market,symbol,モデル名のモデルファイルパスを返す"""
    return os.path.join(get_models_subdir(market, symbol), f"{model_name}.joblib")


# ===== 結果関連パス =====
def get_results_dir() -> str:
    """結果保存用ディレクトリの絶対パスを返す"""
    return os.path.join(_PYTHON_ROOT, "results")


def get_results_subdir(timestamp: str = None) -> str:
    """
    結果サブディレクトリの絶対パスを返す
    
    Args:
        timestamp: タイムスタンプ文字列（例: "20260213_120000"）。Noneの場合は現在日時
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(get_results_dir(), timestamp)


# ===== 設定ファイル関連パス =====
def get_watchlist_path() -> str:
    """データ取得対象CSVのパスを返す"""
    return os.path.join(_PYTHON_ROOT, "データ取得対象.csv")


def get_monitor_list_path() -> str:
    """監視対象CSVのパスを返す"""
    return os.path.join(_PYTHON_ROOT, "監視対象.csv")


# ===== ティッカー補正 =====
def get_ticker(market: str, symbol: str) -> str:
    """
    市場ごとにティッカーを補正して返す（例：日本株は .T を付与）
    """
    market_lower = market.lower()
    if market_lower in ["jp", "japan"]:
        if symbol.endswith(".T"):
            return symbol
        else:
            return f"{symbol}.T"
    elif market_lower in ["us", "usa", "nyse", "nasdaq"]:
        return symbol
    # 他市場は必要に応じて拡張
    return symbol


# ===== ユーティリティ =====
def ensure_dir(path: str) -> str:
    """ディレクトリが存在しなければ作成し、パスを返す"""
    os.makedirs(path, exist_ok=True)
    return path
