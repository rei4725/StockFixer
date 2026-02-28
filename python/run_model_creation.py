"""
銘柄別モデル作成スクリプト

単一銘柄: py run_model_creation.py --market us --symbol AAPL
バッチ:   py run_model_creation.py --batch
"""

import argparse
from src.services.model_training_pipeline import train_models_for_symbol, train_models_for_symbol_task


# バッチ作成の並列数（CPU数に応じて調整）
MAX_MODEL_WORKERS = 4


def run_single(market: str, symbol: str):
    """単一銘柄のモデルを作成する"""
    result = train_models_for_symbol(market, symbol)
    if result["status"] == "success":
        print(f"{market}/{symbol} のモデル作成が完了しました。")
    elif result["status"] == "skip":
        print(f"{market}/{symbol}: {result.get('reason', 'スキップ')}")
    else:
        print(f"{market}/{symbol} のモデル作成に失敗: {result.get('error', '不明')}")


def run_batch():
    """ウォッチリストの全銘柄を並列でモデル作成する"""
    from src.services.batch_runner import load_target_symbols, run_parallel, print_summary

    symbols = load_target_symbols()
    if not symbols:
        print("対象銘柄がありません。")
        return

    results = run_parallel(
        func=train_models_for_symbol_task,
        tasks=symbols,
        max_workers=MAX_MODEL_WORKERS,
        use_process=True,
        label="モデル作成",
    )
    print_summary("モデル作成", results)


def main():
    parser = argparse.ArgumentParser(description="銘柄別モデル作成スクリプト")
    parser.add_argument("--batch", action="store_true", help="ウォッチリストの全銘柄を並列で作成する")
    parser.add_argument("--market", type=str, help="市場名（例: us, jp）")
    parser.add_argument("--symbol", type=str, help="銘柄コード（例: AAPL, 7203）")
    args = parser.parse_args()

    if args.batch:
        run_batch()
    else:
        if not args.market or not args.symbol:
            parser.error("--market と --symbol は必須です（--batch 指定時を除く）")
        run_single(args.market, args.symbol)


if __name__ == "__main__":
    main()
