"""
PoC: OllamaClient の動作確認スクリプト

使い方:
    cd python/
    py run_llm_sentiment_poc.py

事前に Ollama コンテナが起動済みで、モデルが pull されている必要がある:
    docker-compose up -d ollama
    docker exec ollama ollama pull qwen2.5:3b
"""

import sys

sys.path.insert(0, "src")

from src.market_data.adapters.llm_sentiment import OllamaClient  # noqa: E402


def main() -> None:
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    client = OllamaClient()

    print(f"Ollama URL : {client._base_url or '(未設定)'}")
    print(f"Model      : {client._model}")
    print(f"Available  : {client.is_available}")
    print()

    if not client.is_available:
        print("[SKIP] Ollama に接続できません。コンテナが起動しているか確認してください。")
        print("  docker-compose up -d ollama")
        print("  docker exec ollama ollama pull qwen2.5:3b")
        return

    test_cases = [
        {
            "label": "強気ニュース (英語)",
            "texts": [
                "Apple beats earnings expectations with record quarterly revenue",
                "Apple stock surges on strong iPhone sales and bullish guidance",
            ],
        },
        {
            "label": "弱気ニュース (英語)",
            "texts": [
                "Company misses earnings forecast, cuts annual outlook",
                "Stock plunges as CEO resigns amid accounting scandal",
            ],
        },
        {
            "label": "中立ニュース",
            "texts": ["Company holds annual shareholder meeting"],
        },
        {
            "label": "日本語開示",
            "texts": [
                "第3四半期決算発表：売上高が前年同期比15%増、営業利益も過去最高を更新",
                "通期業績予想を上方修正、配当も増額予定",
            ],
        },
    ]

    for case in test_cases:
        score = client.score(case["texts"])
        print(f"[{case['label']}]")
        for t in case["texts"]:
            print(f"  テキスト: {t}")
        print(f"  スコア  : {score}")
        print()


if __name__ == "__main__":
    main()
