import os

def get_data_dir():
    """データ保存用ディレクトリの絶対パスを返す"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

def get_models_dir():
    """モデル保存用ディレクトリの絶対パスを返す"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))

def get_data_subdir(market: str, symbol: str):
    """指定market,symbolのデータサブディレクトリの絶対パスを返す"""
    return os.path.join(get_data_dir(), f"{market}_{symbol}")

def get_models_subdir(market: str, symbol: str):
    """指定market,symbolのモデルサブディレクトリの絶対パスを返す"""
    return os.path.join(get_models_dir(), f"{market}_{symbol}")

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
