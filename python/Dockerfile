# StockFixer Python Environment
FROM python:3.12-slim

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
      org.opencontainers.image.source="https://github.com/rei4725/StockFixer" \
      org.opencontainers.image.sbom="https://github.com/rei4725/StockFixer/releases/download/v${VERSION}/sbom.json"

# 作業ディレクトリの設定
WORKDIR /app

# システム依存パッケージのインストール（LightGBM/XGBoost用、postgresql-clientはpg_dumpバックアップ用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のインストール（キャッシュ効率化のため先にコピー）
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade "pip>=25.3" && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir --upgrade \
        "jaraco.context>=6.1.0" \
        "wheel>=0.46.2" && \
    rm /tmp/requirements.txt

# Claude Code CLI（LLM_BACKEND=cli でサブスク認証経由の LLM 呼び出しに使用）
# node 不要のネイティブインストーラで /root/.local/bin に導入する。
# サブスク認証は CLAUDE_CODE_OAUTH_TOKEN（claude setup-token で発行）を env で渡す。
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://claude.ai/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"

# アプリケーションコードのコピー
COPY . .

# 環境変数の設定
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ヘルスチェック（/health エンドポイントを叩いて DB 接続と業務状態を確認）
# timeout はバッチと I/O 競合中の応答遅延（#553 実測: ok パスで最大 11 秒超）を
# 吸収できる値にする。/health 自身が busy 打ち切り（2秒×1接続）で応答を返すため、
# 30 秒あれば十分な余裕がある。
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; r=urllib.request.urlopen('http://localhost:5100/health',timeout=25); sys.exit(0 if r.status==200 else 1)" || exit 1

# デフォルトコマンド: スケジューラ + Discord Bot
CMD ["python", "run_scheduler.py", "--with-bot"]
