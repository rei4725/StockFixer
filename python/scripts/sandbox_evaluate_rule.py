"""サンドボックスコンテナ内で実行される、Claude生成ルールの隔離バックテスト実行スクリプト。

このスクリプトは stockfixer イメージから `docker run --network none` 経由でのみ
起動される想定であり、ホスト側の信頼されたプロセスから直接importして呼ばれることはない。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback

import pandas as pd


def _load_data_by_symbol(data_dir: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".parquet"):
            continue
        symbol = name[: -len(".parquet")]
        data[symbol] = pd.read_parquet(os.path.join(data_dir, name))
    return data


def _finite_or_none(value: float) -> float | None:
    """JSON に NaN を書かない（ホスト側は None を「算出不能」として扱う）。"""
    return None if value is None or math.isnan(value) else float(value)


def _series_payload(returns: pd.Series | None) -> dict | None:
    """ポートフォリオ日次リターンを JSON で運べる形にする。"""
    if returns is None:
        return None
    return {
        "dates": [ts.isoformat() for ts in returns.index],
        "values": [float(v) for v in returns.to_numpy()],
    }


def _load_windows(windows_file: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    with open(windows_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(pd.Timestamp(w[0]), pd.Timestamp(w[1])) for w in raw]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--rule-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--lookback-years", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--windows-file", required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()

    if os.environ.get("STOCKFIXER_SANDBOX") != "1":
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "EnvironmentError",
                    "traceback": "STOCKFIXER_SANDBOX=1 が設定されていません。"
                    "このスクリプトはサンドボックスコンテナ内専用です。",
                }
            )
        )
        return 1

    # 入力が見えないのは候補コードの誤りではなく受け渡しの問題（#757）。
    # 実行時エラー（修復対象）と区別して報告し、ホスト側が Claude に修復を頼まないようにする
    missing = [
        path
        for path, exists in (
            (args.source_file, os.path.isfile(args.source_file)),
            (args.data_dir, os.path.isdir(args.data_dir)),
            (args.windows_file, os.path.isfile(args.windows_file)),
        )
        if not exists
    ]
    if missing:
        print(json.dumps({"status": "input_missing", "detail": f"見つからない入力: {missing}"}))
        return 1

    from src.backtest.factory import build_rule, evaluate_hypothesis
    from src.backtest.types import FactoryHypothesis

    try:
        with open(args.source_file, "r", encoding="utf-8") as f:
            source_code = f.read()

        hypothesis = FactoryHypothesis(
            rule_spec={
                "type": "generated_code",
                "source_code": source_code,
                "class_name": args.class_name,
                "rule_name": args.rule_name,
                "description": args.description,
            },
            market=args.market,
            lookback_years=args.lookback_years,
        )
        data_by_symbol = _load_data_by_symbol(args.data_dir)
        windows = _load_windows(args.windows_file)

        # evaluate_hypothesis() は銘柄ごとの例外を握りつぶし「取引数0」として扱うため、
        # generate_signal() のクラッシュがそのままではゲート不合格（修復不能）に埋もれて
        # しまう（Task 7の実機結合テストで発覚）。先に1銘柄分だけ直接呼び、クラッシュを
        # 「修復可能」な失敗として正しく検出できるようにする。
        if data_by_symbol:
            smoke_symbol = sorted(data_by_symbol)[0]
            smoke_rule = build_rule(hypothesis.rule_spec)
            smoke_rule.generate_signal(data_by_symbol[smoke_symbol])

        evaluation = evaluate_hypothesis(hypothesis, data_by_symbol, windows)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "evaluation": {
                        "sharpe_ratio": evaluation.sharpe_ratio,
                        "sharpe_per_trade": evaluation.sharpe_per_trade,
                        "win_rate": evaluation.win_rate,
                        "num_trades": evaluation.num_trades,
                        "max_drawdown": evaluation.max_drawdown,
                        "total_return": evaluation.total_return,
                        "window_returns": evaluation.window_returns,
                        "n_symbols": evaluation.n_symbols,
                        "n_symbols_with_signal": evaluation.n_symbols_with_signal,
                        "n_effective_symbols": evaluation.n_effective_symbols,
                        "avg_trades_per_symbol": evaluation.avg_trades_per_symbol,
                        # ゲートはポートフォリオ日次リターンで判定するため、ホスト側で
                        # チャンピオンとの検定・DSR ができるよう系列ごと返す
                        "portfolio_sharpe_ratio": _finite_or_none(
                            evaluation.portfolio_sharpe_ratio
                        ),
                        "portfolio_max_drawdown": _finite_or_none(
                            evaluation.portfolio_max_drawdown
                        ),
                        "portfolio_returns": _series_payload(evaluation.portfolio_returns),
                    },
                }
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
