# 長期バックテストのクラス分割設計（Phase 2）

作成日: 2026-09-21
対象 BC: `src/backtest/`（`longterm_backtest.py` および `metrics/`）
先行 spec: `2026-09-21-execution-model-design.md`（Phase 1 / PR #730 マージ済）

## 背景

Phase 1 で取引コストを `ExecutionModel` に集約した結果、longterm エンジンの残る
問題が構造の側に絞り込まれた。

| 症状 | 現状 |
|---|---|
| 引数の氾濫 | `run_longterm_backtest` が 11 引数、`_try_enter` が 9 引数。CLI 側と既定値が二重管理 |
| Types over dicts 違反 | `open_positions: dict[str, dict[str, Any]]`。7 キーの生 dict で、`pos["current_hf"]` は型検査を素通りする |
| 状態が手続きに散る | `cash` を `_try_enter` の引数と戻り値で往復させる |
| 単一銘柄バックテストと比較不能 | `_compute_metrics` が `metrics/core` と並立し、sharpe / profit_factor が無い |
| リスクリーン当日 Close 約定 | `Backtester` は `#493` で `execution_lag=1` を既定にしたが、longterm には適用されていない |
| 結論文が嘘をつく | `build_conclusion` が `rescreen_freq` も `HoldRules` も受け取らないのに「四半期スクリーン」「40 週線割れ」を文字列リテラルで埋めている |

## スコープ

**対象は longterm エンジンのみ。** `portfolio/simulation.py` の
`holdings: dict[str, dict[str, Any]]` も同種の生 dict だが、Phase 2 では扱わない。
別 Issue とする。理由は、差分を 1 エンジンに閉じることでテストの patch 経路の
付け替えを限定でき、振る舞い変更の帰属が明確になるためである。

`screening/hold_engine.py` には手を入れない。本番のスクリーニング出力が共用して
おり、Phase 3（ポート化）の前に侵入すると順序が逆になる。

## 検討した代替案

**案B: Config のみ立て、保有は `OpenPosition` の dataclass 化まで。** 差分は小さいが
`cash` の受け渡しが残り、`_try_enter` の引数は 6 個程度にしか減らない。さらに
`execution_lag` 用のカレンダー索引を追加で渡すことになり、引数削減が相殺される。

**案D: 日次ステップ化。** `simulate_position` はエントリー時点で当該銘柄の全未来の
イベント列を一括生成し、`events_by_date` に台本として格納する。日次ループはそれを
再生しているだけである。この構造ではポートフォリオ状態に依存する規則（現金不足時の
見送り、セクター上限、ポートフォリオ・ボラでのサイジング）が表現できない。案D は
台本を解体し、各 `OpenPosition` が日々ルール状態を更新する形に変える。

採用しない。逐次版の hold engine が必要になるが、`hold_engine.py` は screening BC に
あり本番のスクリーニング出力が使っている。また「ポートフォリオ状態に依存する規則」は
現時点で誰も要求しておらず、YAGNI に反する。

**案E + 案F（採用）: 取引台帳を敷き、整数株に揃える。** `Portfolio` が全売買を 1 本の
イベント列として追記する。台帳の役割は 3 つである。

1. 監査証跡
2. 「現金 + 保有評価 == 台帳から再構成した equity」という不変条件をテストで書ける
   根拠（`#483` の過剰買付と同系統の事故を機械的に封じる）
3. 銘柄・数量を取り違えない正しい損益列の出所

あわせて端株をやめ整数株にする（`max_affordable_qty`）。Phase 1 で「コスト以外の
振る舞い変更は範囲外」として送った判断の、送り先が本 Phase である。

## 設計

### モジュール構成

```
src/backtest/longterm/
├── __init__.py      公開 API の再輸出
├── config.py        LongtermBacktestConfig（frozen）
├── portfolio.py     OpenPosition / ClosedTrade / Portfolio
├── ledger.py        TradeRecord / TradeLedger
├── prices.py        価格ロード・カレンダー・リスクリーン日・close 索引
├── engine.py        run_longterm_backtest 本体 + enter_candidates
├── metrics.py       longterm 固有の倍率系 + stats の呼び出し
└── reporting.py     save_results / build_conclusion

src/backtest/metrics/stats.py        新設（純粋統計）
src/backtest/longterm_backtest.py    re-export ファサード
```

ファサードを残すのは `run_longterm_backtest.py` の import 互換のためである。ただし
**ファサードへの `patch.object` は `engine.py` に届かない**（engine が自前で import
するため）。テストの patch 先は `lb.engine` へ付け替える。ファサードが保証するのは
`run_longterm_backtest` / `build_conclusion` / `save_results` の呼び出し互換のみと
割り切る。これは Phase 1 で 3 ファイル分発生したのと同じ機械作業である。

### 型

```python
# longterm/config.py
@dataclass(frozen=True)
class LongtermBacktestConfig:
    market: str = "us"
    start: str = "2021-01-01"
    end: str = "2026-01-01"
    rescreen_freq: str = "quarterly"
    top_n: int = 30
    initial_cash: float = 1_000_000.0
    max_positions: int = 10
    execution_lag: int = 1
    benchmark_ticker: str = "^GSPC"
    n_trials: int = 0
    costs: TradingCosts = field(default_factory=TradingCosts)
    rules: HoldRules = field(default_factory=HoldRules)

    @classmethod
    def build(cls, *, market: str, fee_rate: float = DEFAULT_FEE_RATE,
              slippage: float | None = None, **kw) -> "LongtermBacktestConfig":
        """市場別コスト解決を含む組み立て。CLI はこれを呼ぶ。"""
```

```python
# longterm/portfolio.py
@dataclass
class OpenPosition:
    symbol: str
    entry_date: str
    entry_price: float
    shares: int            # 案F: 整数株
    current_hf: float = 1.0
    cost_basis: float = 0.0   # 手数料・スリッページ込み取得原価（損益の正本）
    realized: float = 0.0     # 部分利確の累計受取額
    events_by_date: dict[str, list[PositionEvent]] = field(default_factory=dict)

    def held_shares(self) -> float: ...
    def market_value(self, price: float) -> float: ...


@dataclass(frozen=True)
class ClosedTrade:
    symbol: str; entry_date: str; exit_date: str
    entry_price: float; exit_price: float
    multiple: float; max_multiple: float
    exit_reason: str; held_days: int
    realized_pnl: float; return_rate: float


class Portfolio:
    cash: float
    positions: dict[str, OpenPosition]
    ledger: TradeLedger
    closed: list[ClosedTrade]

    def enter(self, pos: OpenPosition, execution: ExecutionModel) -> None: ...
    def scale_out(self, symbol: str, ev: PositionEvent, execution: ExecutionModel) -> None: ...
    def close(self, symbol: str, ev: PositionEvent, execution: ExecutionModel,
              price_map: dict[str, pd.DataFrame]) -> ClosedTrade: ...
    def equity(self, close_lookup: dict[str, dict[str, float]], date: str) -> float: ...
```

`Portfolio` は可変である。frozen にすると日次ループ約 1200 回 × 保有銘柄数のコピーが
走るため、ここは可変が正しい。

`_try_enter` は `enter_candidates(config, portfolio, date, price_map, execution)` に
なり、引数 9 個から 5 個に減る。`cash` の往復は `Portfolio` の内部状態になって消える。

### メトリクスの繋ぎ方

**`core.compute_metrics` は呼ばない。** `_extract_trade_pnl`（`metrics/core.py:210`）は
buy / sell を FIFO で対にするが、

```python
buy = buys.pop(0)
pnl = (row["price"] - buy["price"]) * buy["qty"]
```

- **銘柄を見ていない。** 単一銘柄の `Backtester` では正しいが、longterm は最大 10
  銘柄を同時保有する。ある銘柄の売りが別銘柄の買いと対になり損益が誤る。
- **数量を合わせていない。** longterm は部分利確を持つため、一部売却が買い 1 件を
  丸ごと消費する。

`core` 側を修正する道は採らない。`compute_metrics` の `sharpe_ratio` は戦略
ファクトリーの合格ゲートが読んでいる（`#719` で是正した判定基準の入力）。ここを
動かすと過去の合否判定と接続できなくなる。

代わりに**統計関数のみを共有する**。

```
src/backtest/metrics/stats.py（新設・純粋関数）
    max_drawdown(equity)           core._max_drawdown を移設
    sharpe_per_trade(pnls, rf)     core._sharpe_per_trade を移設
    annualize_sharpe(spt, tpy)     core._annualize_sharpe を移設
    profit_factor(wins, losses)    core のインライン式を抽出
    cagr(initial, final, years)    新規（core には無い）
```

`core.py` は `from src.backtest.metrics.stats import ...` に置換し、私有名は別名として
残す。**計算式は一切変更しない。** よって `Backtester` の数値ピン止め 20 件も
ファクトリーのゲートも動かない。

longterm 側は損益列を自前で作る。`Portfolio` は建値・株数・部分利確を保持しており、
決済時に `ClosedTrade.realized_pnl` を正しく吐ける。銘柄の取り違えも数量ずれも構造上
起こらない。

**DSR は既定では計算されない。** `deflated_sharpe_ratio` は `n_trials > 0` を要する。
longterm はグリッドサーチではなく試行回数の概念がないため、`config.n_trials` を
任意で開けておき既定 0 = DSR 省略とする。「longterm にも過学習ガードがかかる」とは
主張しない。

### 振る舞い変更の勘定

| # | 変更 | 方向 | 説明可能性の担保 |
|---|---|---|---|
| F-1 | 端株 → 整数株 | 端数分の現金が遊ぶ。わずかに悪化 | 差額 < 1 株分 × 保有銘柄数 |
| F-2 | `qty = 0` になる高値銘柄は見送り | 保有数が減りうる | 境界をテストで固定 |
| L-1 | エントリーが翌営業日 Close | 方向は定まらない | lag=0 との対照実行を PR 本文に並記 |
| L-2 | リスクリーン日が暦末尾なら翌日が無く見送り | 最大 1 件減 | 境界テスト |
| M-1 | `n_trades` → `num_trades` | キー名のみ | CLI 出力とテストを同時に付け替え |
| M-2 | `sharpe_ratio` / `profit_factor` / `calmar_ratio` 追加 | 追加のみ | 既存キーは不変 |
| C-1 | 結論文が `rescreen_freq` と `trail_ma_weeks` を反映 | 文言のみ | `--rescreen-freq weekly` で「週次スクリーン」と出ること |

**受け入れ基準**: PR-4 は既存ユニットテストが数値を一切変えずに通ること。PR-5 は
F-1 の悪化が端株切り捨ての理論上限以内に収まり、L-1 は lag=0/1 の対照を並記する。
桁違いに動いた場合はバグを疑い、マージしない。

### PR 分割

| PR | 内容 | 振る舞い | version_impact |
|---|---|---|---|
| PR-4 | パッケージ分割 + Config / Portfolio / OpenPosition + 台帳 | **不変**（台帳は並走。既存出力を台帳から導出し同値検証） | patch |
| PR-5 | 整数株 + `execution_lag=1` 既定 | 変わる | minor |
| PR-6 | `metrics/stats.py` 抽出 + longterm メトリクス拡充 + `build_conclusion` 修正 | 指標追加・文言 | minor |

PR-4 を振る舞い不変に隔離するのが要である。既存テストが数値を変えずに通ることが、
型分割にバグがないことの唯一の証明になる。そこへ PR-5 で数値を動かせば、差分は lag と
整数株だけに帰属できる。混ぜると「この悪化は型分割のバグか lag か」が事後に判別
できない。

### テスト

新規 `tests/unit/backtest/longterm/`:

| ファイル | 検証 |
|---|---|
| `test_config.py` | `build()` の市場別コスト解決（us / jp / 未知）、既定値 |
| `test_portfolio.py` | `enter` → `scale_out` → `close` の現金推移。現金が負にならないこと。部分利確後の `shares * current_hf` が保有株数と一致 |
| `test_ledger.py` | 不変条件: 任意の日で `cash + 保有評価 == equity_df` の値。台帳の buy 総額 − sell 総額 == 現金の減少分 |
| `test_longterm_metrics.py` | 損益列が銘柄ごとに正しく対になること（`_extract_trade_pnl` の取り違えを再現させない回帰テスト）。`profit_factor` / `sharpe` の健全性 |
| `test_entry_lag.py` | lag=1 で建値が翌営業日 Close。lag=0 で従来値。暦末尾の見送り |
| `test_conclusion.py` | `weekly` で「週次」、`trail_ma_weeks=30` で「30 週線」 |

既存 `tests/unit/test_longterm_backtest.py`（215 行）は `patch.object(lb, ...)` を
`lb.engine` へ付け替える。数値アサートを持たない設計のため、PR-4 では patch 先以外の
書き換えは発生しない見込みである。

`metrics/stats.py` 抽出の安全網は、`test_metrics*.py` と `test_backtester_*.py` が
無改変で通ることとする。

## Phase 2 の対象外

- `portfolio/simulation.py` の `holdings` / `eq_holdings` 生 dict（別 Issue）
- `screening/hold_engine.py` への逐次ステップ化（案D。Phase 3 のポート化後に再検討）
- 撤退・利確側の約定ラグ（同上。`hold_engine.py` にラグ概念の導入が必要）
- `core.compute_metrics` の銘柄非対応 FIFO（`Backtester` は単一銘柄のため実害なし。
  多銘柄の利用者が現れた時点で対処する）
