"""
銘柄別モデル作成スクリプト

単一銘柄: py run_model_creation.py --market us --symbol AAPL
バッチ:   py run_model_creation.py --batch
"""

import argparse
from src.services.model_training_pipeline import train_models_for_symbol


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


def _load_features_task(task: dict) -> dict:
    """バッチランナー用: DB読み込みのみ（並列安全）"""
    from src.services.model_training_pipeline import load_features_for_training
    return load_features_for_training(task["market"], task["symbol"])


def run_batch():
    """
    ウォッチリストの全銘柄のモデルを作成する。

    フェーズ1: DB読み込み（並列） - DuckDB読み取りはスレッド並列で安全
    フェーズ2: モデル学習・保存（逐次） - ファイルI/Oの競合を回避
    """
    from src.services.batch_runner import load_target_symbols, run_parallel, print_summary
    from src.models.model_manager import ModelManager

    symbols = load_target_symbols()
    if not symbols:
        print("対象銘柄がありません。")
        return

    # フェーズ1: データ読み込み（並列）
    load_results = run_parallel(
        func=_load_features_task,
        tasks=symbols,
        max_workers=MAX_MODEL_WORKERS,
        label="データ読み込み",
    )

    # フェーズ2: モデル学習・保存（逐次）
    success_data = [r for r in load_results if r.get("status") == "success" and "X" in r]
    print(f"\n{'='*50}")
    print(f"モデル学習開始（逐次）")
    print(f"対象件数: {len(success_data)}")
    print(f"{'='*50}\n")

    train_results = []
    for i, r in enumerate(success_data, 1):
        market, symbol = r["market"], r["symbol"]
        try:
            print(f"[モデル作成開始] {market}/{symbol}")
            model_manager = ModelManager()
            model_manager.create_model("XGBoostModel", "StockXGBoostModel")
            model_manager.train_model("StockXGBoostModel", r["X"], r["y"], market=market, symbol=symbol)
            model_manager.create_model("LightGBMModel", "StockLightGBMModel")
            model_manager.train_model("StockLightGBMModel", r["X"], r["y"], market=market, symbol=symbol)
            print(f"[モデル作成完了] {market}/{symbol}")
            train_results.append({"market": market, "symbol": symbol, "status": "success"})
        except Exception as e:
            print(f"[モデル作成エラー] {market}/{symbol}: {e}")
            train_results.append({"market": market, "symbol": symbol, "status": "error", "error": str(e)})
        if i % 50 == 0:
            print(f"  ... {i}/{len(success_data)} 件完了")

    # 最終サマリー
    final_results = train_results.copy()
    final_results += [r for r in load_results if r.get("status") in ("error", "skip")]
    print_summary("モデル作成", final_results)


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
