from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, request

# プロジェクト内のモジュールをインポート
from src.data.data_loader import get_stock_data
from src.features.technical_analysis import add_technical_indicators
from src.models.model_manager import ModelManager  # AIモデルの管理クラス
from src.models.predict_single_stock import predict_single_stock
from src.sbi import sbi_api
from src.strategy.signal_generator import SignalGenerator

app = Flask(__name__)

# 各モジュールのインスタンス化
model_manager = ModelManager()
signal_generator = SignalGenerator()


@app.route("/health", methods=["GET"])
def health_check():
    """APIサーバーのヘルスチェックエンドポイント。"""
    return jsonify({"status": "ok", "message": "API server is running."}), 200


@app.route("/generate_signal", methods=["POST"])
def generate_signal():
    """
    指定されたシンボルの最新データに基づいて売買シグナルを生成し返却します。
    リクエストボディ:
    {
        "symbol": "AAPL",
        "market": "us",          # 省略可（デフォルト: us）
        "start_date": "2023-01-01",  # 省略可（RSI計算用ウィンドウ）
        "end_date": "2023-12-31"     # 省略可（RSI計算用ウィンドウ）
    }
    """
    data = request.get_json()
    symbol = data.get("symbol")
    market = data.get("market", "us")
    start_date_str = data.get("start_date")
    end_date_str = data.get("end_date")

    if not symbol:
        return (
            jsonify({"error": "Missing parameter: symbol is required."}),
            400,
        )

    for date_str in [start_date_str, end_date_str]:
        if date_str:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    try:
        # 1. 学習済みモデルで予測実行
        result_df = predict_single_stock(market, symbol)
        if result_df is None:
            return (
                jsonify({"error": f"No trained model found for {market}/{symbol}."}),
                404,
            )

        diff_ratio = float(result_df["diff_ratio"].iloc[0])
        current_price = float(result_df["current_price"].iloc[0])
        pred_price = float(result_df["avg_pred_price"].iloc[0])

        # 2. テクニカル指標を取得（RSI補強シグナル用）
        df = get_stock_data(market, symbol, start_date_str, end_date_str)
        if df is not None and not df.empty:
            df_ta = add_technical_indicators(df.copy())
            today = df_ta.index[-1]
            prediction_series = pd.Series({today: diff_ratio})
            df_for_signal = df_ta.iloc[[-1]]
        else:
            today = datetime.today()
            prediction_series = pd.Series({today: diff_ratio})
            df_for_signal = pd.DataFrame(index=[today])

        # 3. シグナル生成（最新日1行のみ）
        signals = signal_generator.generate_signal(df_for_signal, prediction_series)
        signal_value = str(signals.iloc[0]) if len(signals) > 0 else "Hold"

        result = [
            {
                "date": today.strftime("%Y-%m-%d")
                if hasattr(today, "strftime")
                else str(today)[:10],
                "signal": signal_value,
                "close_price": current_price,
                "predicted_price": pred_price,
                "predicted_change": diff_ratio,
            }
        ]
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Error generating signal: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/sbi_login", methods=["POST"])
def sbi_login():
    """
    SBI証券へログインします。
    リクエストボディ:
    {
        "user_id": "your_user_id",
        "password": "your_password"
    }
    """
    data = request.get_json()
    user_id = data.get("user_id")
    password = data.get("password")

    if not all([user_id, password]):
        return jsonify({"error": "Missing parameters: user_id, password are required."}), 400

    try:
        result, status_code = sbi_api.sbi_login(user_id, password)
        return jsonify(result), status_code

    except Exception as e:
        app.logger.error(f"Error logging in to SBI: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # 開発用サーバーの起動
    # 本番環境ではGunicornなどのWSGIサーバーを使用することを推奨
    app.run(debug=True, host="0.0.0.0", port=5000)
