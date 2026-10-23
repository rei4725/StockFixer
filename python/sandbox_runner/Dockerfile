# python/sandbox_runner/Dockerfile
# docker.sock を持つ唯一のコンテナ。決まった設定のサンドボックス実行だけを引き受ける
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-29.9.0.tgz -o /tmp/docker.tgz \
    && tar xzf /tmp/docker.tgz -C /tmp && mv /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker.tgz /tmp/docker /var/lib/apt/lists/*
WORKDIR /runner
COPY sandbox_runner/ sandbox_runner/
ENV PYTHONUNBUFFERED=1
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8700/health',timeout=4).status==200 else 1)"
CMD ["python", "-m", "sandbox_runner.server"]
