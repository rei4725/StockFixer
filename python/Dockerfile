# StockFixer Python Environment
FROM python:3.11-slim

# 作業ディレクトリの設定
WORKDIR /app

# システム依存パッケージのインストール（LightGBM/XGBoost用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のインストール（キャッシュ効率化のため先にコピー）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードのコピー
COPY . .

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# デフォルトコマンド（必要に応じて変更）
CMD ["python", "run_discord_bot.py"]
