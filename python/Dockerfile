# StockFixer Python Environment
FROM python:3.11-slim

# ビルド引数（イメージメタデータ）
ARG BUILD_DATE=unknown
ARG GIT_COMMIT=unknown
ARG VERSION=dev

# OCI 標準ラベル
LABEL org.opencontainers.image.title="StockFixer" \
      org.opencontainers.image.description="リアルタイム自動売買システム - Python環境" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.source="https://github.com/rei4725/StockFixer"

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
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ヘルスチェック（プロセス生存確認）
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python -c "import os, signal; os.kill(1, signal.SIG_DFL)" || exit 1

# デフォルトコマンド: スケジューラ + Discord Bot
CMD ["python", "run_scheduler.py", "--with-bot"]
