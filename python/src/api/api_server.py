from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# プロジェクト内のモジュールをインポート
from src.data.data_loader import get_stock_data
from src.features.technical_analysis import add_technical_indicators
from src.models.model_manager import ModelManager # AIモデルの管理クラス
from src.strategy.signal_generator import SignalGenerator
from src.sbi.sbi_api import sbi_api

app = Flask(__name__)

# 各モジュールのインスタンス化
model_manager = ModelManager()
signal_generator = SignalGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """
    APIサーバーのヘルスチェックエンドポイント。
    """
    return jsonify({"status": "ok", "message": "API server is running."}), 200

@app.route('/generate_signal', methods=['POST'])
def generate_signal():
    """
    指定されたシンボルと期間のデータに基づいて売買シグナルを生成し返却します。
    リクエストボディ:
    {
        "symbol": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }
    """
    data = request.get_json()
    symbol = data.get('symbol')
    start_date_str = data.get('start_date')
    end_date_str = data.get('end_date')

    if not all([symbol, start_date_str, end_date_str]):
        return jsonify({"error": "Missing parameters: symbol, start_date, end_date are required."}), 400

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    try:
        # 1. データのロード
        df = get_stock_data(symbol, start_date_str, end_date_str)
        if df.empty:
            return jsonify({"message": f"No data found for {symbol} from {start_date_str} to {end_date_str}."}), 404

        # 2. テクニカル分析の実行
        df_ta = add_technical_indicators(df.copy())

        # 3. AI予測の実行 (ダミー予測として、ここではランダムな値を生成)
        # 実際には、ここで学習済みモデルをロードし、予測を実行します。
        # 例: prediction = model_manager.predict(df_ta)
        # ここでは、Close価格の翌日変動率を予測していると仮定
        prediction = pd.Series(np.random.uniform(-0.01, 0.01, len(df_ta)), index=df_ta.index)

        # 4. シグナル生成
        signals = signal_generator.generate_signal(df_ta, prediction)

        # 結果をJSON形式で返す
        result = []
        for date, signal in signals.items():
            result.append({
                "date": date.strftime('%Y-%m-%d'),
                "signal": signal,
                "close_price": df.loc[date, 'Close'] if date in df.index else None,
                "predicted_change": prediction.loc[date] if date in prediction.index else None
            })
        
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Error generating signal: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/sbi_login', methods=['POST'])
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
    user_id = data.get('user_id')
    password = data.get('password')

    if not all([user_id, password]):
        return jsonify({"error": "Missing parameters: user_id, password are required."}), 400

    try:
        result = sbi_api.sbi_login(user_id, password)
        return jsonify({"result": result}), 200

    except Exception as e:
        app.logger.error(f"Error logging in to SBI: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 開発用サーバーの起動
    # 本番環境ではGunicornなどのWSGIサーバーを使用することを推奨
    app.run(debug=True, host='0.0.0.0', port=5000)
