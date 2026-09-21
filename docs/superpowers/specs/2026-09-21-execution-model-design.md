# ExecutionModel 導入設計（Phase 1 / PR-1）

作成日: 2026-09-21
対象 BC: `src/backtest/`
version_impact: minor（振る舞いが変わるため patch ではない）

## 背景

`longterm_backtest.py` を題材にした4観点レビュー（reuse / simplification /
efficiency / altitude）の診断が一点に収束した。新しいエンジンを足す判断自体は
正しいが、足す際に既存の正本（`ports.py` 機構・`metrics/core`・
`pipeline/reporting`・`config/settings.py` のコスト定数・`execution_lag` 規律）を
どれ一つ通らず、全部を自前で薄く再実装していた。

その中で最も影響が大きいのが取引コストの扱いである。手数料演算は backtest BC 内の
21 箇所に手書きで散っている。

| ファイル | 手書き箇所 |
|---|---|
| `src/backtest/backtester.py` | 11 |
| `src/backtest/portfolio/simulation.py` | 6 |
| `src/backtest/longterm_backtest.py` | 3 |
| `src/backtest/position_sizing.py` | 1 |

散在の結果、次の不整合が生じている。

1. **スリッページが片方のエンジンにしか届かない。** `#494` の市場別既定
   （`DEFAULT_SLIPPAGE_US = 0.0005` / `DEFAULT_SLIPPAGE_JP = 0.0010`）と `R-210` の
   動的スリッページ（`slippage.py` の平方根インパクトモデル）を通るのは
   `Backtester` だけである。`portfolio/simulation.py` にはスリッページの概念が
   そもそも無く、`longterm_backtest.py` も `fee_rate` のみを持つ。
2. **手数料の掛け方の式が揃っていない。** `longterm_backtest.py` は
   `shares = per_position * (1 - fee) / price`（予算から手数料を控除）だが、他は
   `cost = qty * price * (1 + fee)`（価格に上乗せ）である。同じ `fee_rate=0.001`
   でも約定株数がずれる。
3. **既定値が二重管理されている。** `longterm` は `fee_rate=0.001` をエンジン既定と
   CLI 既定の 2 箇所に別々にハードコードし、`pipeline/runner.py` の
   `default_slippage_for(market)` という市場別解決の正本を通らない。

結果として、`longterm` の数字は単一銘柄バックテストと**比較不能**な楽観値になって
いる。

### スコープ外

`src/trading/regime_leverage_strategy/service.py` の `_calc_commission_usd` は実
ブローカーの段階制手数料（USD → JPY 換算）であり、概念が別物のため対象としない。
`src/trading/brokers/paper/paper_broker.py` には手数料の概念が無い。手数料演算の
利用者は backtest BC 内に閉じているため、`ExecutionModel` を `src/domain/` に
上げる理由は無い。

## 検討した代替案

**案A: `TradingCosts` 値オブジェクトのみ。** 21 箇所の手書きを置換するが、動的
スリッページ（`slippage_fn`）は `Backtester` に残す。差分は機械的で読みやすいが、
`#494` の成果物が片方のエンジンに届かないままで、二重構造が残る。

**案C: 約定ラグまで `ExecutionModel` に入れる。** 採用しない。`Backtester` のラグは
「シグナル系列を `shift(n)` して翌バーの Open で約定」というバー配列操作だが、
`longterm` はイベント駆動でカレンダーを歩く。同じ型に押し込めても実装は共有できず、
抽象だけが太る。約定ラグはエンジン側の責務に残す。

**案B（採用）: `ExecutionModel` クラス。** `TradingCosts` を内包し、動的
スリッページの合成規則も引き取る。両エンジンが同じ型を持つ。

## 設計

### 型と API

`src/backtest/execution.py`（新規、約 120 行）に 2 つの型を置く。

```python
@dataclass(frozen=True)
class TradingCosts:
    fee_rate: float = 0.0
    slippage_rate: float = 0.0

    @classmethod
    def for_market(cls, market: str, fee_rate: float = 0.001,
                   slippage: float | None = None) -> "TradingCosts":
        """slippage 未指定なら市場別既定を使う。

        config/settings.py の DEFAULT_SLIPPAGE_US / _JP を正本として直接参照する。
        """


class ExecutionModel:
    def __init__(self, costs: TradingCosts,
                 slippage_fn: Callable[[int, float, float], float] | None = None) -> None: ...

    def effective_slippage(self, qty: int, price: float, volume: float = 0.0) -> float:
        """base + 動的インパクト。Backtester._get_slippage をそのまま移設する。"""

    def unit_buy_cost(self, price: float, volume: float = 0.0, qty: int = 0) -> float:
        """1 株あたり取得原価 price * (1 + fee + slip)。"""

    def buy_cost(self, qty: int, price: float, volume: float = 0.0) -> float: ...
    def sell_proceeds(self, qty: float, price: float, volume: float = 0.0) -> float: ...
    def max_affordable_qty(self, cash: float, price: float, volume: float = 0.0) -> int: ...
```

API をこの 5 つに絞った根拠は、21 箇所が実際に取っている形がこの 5 パターンしか
無いことである。

| 既存の式 | 出現 | 対応メソッド |
|---|---|---|
| `qty * price * (1 - fee - slip)` | 9 | `sell_proceeds` |
| `qty * price * (1 + fee + slip)` | 4 | `buy_cost` |
| `price * (1 + fee_rate + slippage)` | 1 (`position_sizing.py:45`) | `unit_buy_cost` |
| `int(budget / (price * (1 + fee)))` | 2 (`simulation.py:148,179`) | `max_affordable_qty` |
| `base + slippage_fn(...)` | 1 (`backtester.py:_get_slippage`) | `effective_slippage` |

`sell_proceeds` の `qty` だけ `float` なのは、`longterm` が部分利確
（`original_shares * current_hf`）で端株を持つためである。

**依存の向き**: `execution.py` は `config.settings` のみを参照し、
`pipeline/runner.py` を参照しない。逆向きにすると、`runner.py` が
`ExecutionModel` を組むために `execution.py` を import するため循環参照になる。
既存の `default_slippage_for(market)`（`pipeline/runner.py:20`）は
`TradingCosts.for_market` に吸収し、`runner.py` 側は `execution.py` から
import して使う。`_resolve_slippage`（同 :25）も同様に
`TradingCosts.for_market` + `ExecutionModel` の組み立てに置き換わる。

空売り（`backtester.py:248-251`）は `entry * (1 - fee - slip) - exit * (1 + fee + slip)`
と両メソッドの組合せで表せるため、`short_*` メソッドは足さない。

### 呼び出し側の移行

| 移行先 | やること | 振る舞い |
|---|---|---|
| `backtester.py` | `__init__` は `fee_rate` / `slippage` / `slippage_fn` の 3 引数を受け続ける。内部で `self.execution = ExecutionModel(...)` を組み、`_get_slippage` は `self.execution.effective_slippage` への薄い委譲として残す | 不変 |
| `position_sizing.py` | `fee_rate`, `slippage` の 2 引数を `execution: ExecutionModel` 1 つに置換。呼び出しは `Backtester` 内の 1 箇所のみ | 不変 |
| `portfolio/simulation.py` | `fee_rate: float` 引数を `execution: ExecutionModel` に置換。6 箇所を置換 | 変わる |
| `longterm_backtest.py` | `fee_rate` を `execution` に置換。3 箇所を置換し、式を `buy_cost` / `sell_proceeds` に統一 | 変わる |
| `pipeline/runner.py` | `default_slippage_for` / `_resolve_slippage` を `execution.py` 側へ移し、import に差し替える。`run_backtest_single` の引数は変えない | 不変 |

`Backtester` の引数を後方互換に保つのは、呼び出し元が `pipeline/runner.py`・
`walk_forward.py`・`optimizer/` 3 種・`rule_backtester.py` と広く、同時に触ると
PR が Phase 1 の範囲を超えるためである。`ExecutionModel` を直接受け取る形への
移行は Phase 2 以降に送る。

### 振る舞い変更の範囲

変わるのは 2 エンジンのみで、方向は必ず悪化する（コストが増えるため）。

| エンジン | 変更前 | 変更後 | 悪化要因 |
|---|---|---|---|
| `portfolio/simulation.py` | fee 0.1% のみ | fee + 市場別スリッページ（片道 us 0.05% / jp 0.10%） | リバランス毎に往復。weekly なら年 52 回 × 2 |
| `longterm_backtest.py` | fee 0.1%、式が予算控除型 | fee + スリッページ、式は上乗せ型に統一 | 売買回数が少ない（四半期 × 最大 10 銘柄）ため影響は限定的 |

**差分レポートを PR 本文に添える。** 実 DB で旧実装と新実装を同条件で走らせ、
`total_return` / `cagr` / `max_drawdown` / `sharpe_ratio` / `n_trades` を並べた表を
貼る。これが無いと、悪化が意図通りかバグかを後から判別できない。生成は使い捨て
スクリプトで行い、リポジトリには残さない。

**受け入れ基準**: 悪化幅が「往復スリッページ率 × 売買回数」の概算から説明可能な
範囲に収まること。桁違いに悪化した場合はバグを疑い、マージしない。

### テスト

新規 `tests/unit/backtest/test_execution_model.py`:

- `effective_slippage` — `slippage_fn` の有無、`volume <= 0`、`qty = 0` の各分岐。
  `Backtester._get_slippage` の既存テストと同じケースを型の側に移す
- `buy_cost` / `sell_proceeds` の往復 — `sell_proceeds(q, p) < q * p < buy_cost(q, p)`
  の不等式。コストがゼロなら厳密一致
- `max_affordable_qty` — 買った後に現金が負にならないこと
  （`buy_cost(n, p) <= cash < buy_cost(n + 1, p)`）。これは既存の
  `int(budget / ...)` が持っていない保証であり、過剰買付（`#483` の系譜）の
  再発防止になる
- `TradingCosts.for_market` — us / jp / 未知 market の既定値解決

回帰の安全網:

- `Backtester` 側は `test_backtester_unit.py` と `test_backtester_slippage.py` の
  数値ピン止め 20 件がそのまま通ることを不変性の証明とし、PR のゲートにする
- `portfolio` / `longterm` 側は `test_portfolio_backtest.py`(515 行) と
  `test_longterm_backtest.py`(215 行) が通ること。両者は数値ピン止めのアサートを
  1 件も持たず、関係性と構造で検証しているため、振る舞いが動いても壊れない

## Phase 1 の残りの PR（本 spec の対象外）

**PR-2: CSV 保存ヘルパー。** `ensure_dir(results/<subdir>)` + タイムスタンプ +
`to_csv` + パス返却という骨格が 5 箇所（`portfolio/reporting.py`,
`longterm_backtest.py`, `optimizer/persistence.py`, `stress_test.py`,
`screening/trend_screener.py`）にある。ただし差異が 4 つあり、統合には判断が要る。
タイムゾーン（`longterm` と `trend_screener` は UTC、他 3 つはローカル時刻）、
エンコーディング（`stress_test` のみ `utf-8-sig`）、出力本数（1 本 or 2 本）、通知
（`print` / `logger` / なし）。CLAUDE.md の「内部 UTC」が正本なので UTC に寄せるのが
筋だが、既存出力のファイル名が 9 時間ずれる。

**PR-3: リバランス暦と価格行列ローダー。** `portfolio/simulation.py` の
`_get_rebalance_dates`（daily/weekly/monthly）に quarterly/yearly を足して
`longterm` の `_make_rescreen_dates` と共用する。あわせて
`src/utils/db/market_data.py` に `load_all_raw_ohlcv` の一括クエリを追加し、
`_load_price_map` の N+1 クエリ（500 銘柄 = 500 クエリ）を 1 クエリにする。

価格行列の ffill は共通化しない。3 箇所は fill ポリシーが異なる別物である
（`portfolio/simulation.py:244` は `sort_index().ffill()`、
`factory_portfolio.py:48` は `fillna(_CASH_LEVEL)` を足し、
`trading/paper_equity.py:126` は `bfill()` を足す。最後のものは先読みを含む）。
共通化できるのは DB から close 行列を作るローダー部分までであり、fill は呼び出し側の
判断として残す。

## 後続 Phase（参考）

- **Phase 2**: BC 内部のクラス分割。`LongtermBacktestConfig` / `Portfolio` /
  `OpenPosition` を立てる（`_try_enter` の引数 11 個、`open_positions` の生 dict は
  CLAUDE.md の「Types over dicts」違反）。
- **Phase 3**: ポート化。`backtest → screening` の `ignore_imports` 6 行
  （`.importlinter` の Contract 1 と 2 に各 3 行）を消す。既存の
  `backtest/ports.py`・`data_port.py` と同じ書式、合成ルートは
  `orchestration/port_wiring.wire_ports()`。
- **Phase 4**: パッケージ / リポジトリ分割は行わない。デプロイ単位は `stockfixer`
  コンテナ 1 つで、物理分割はバージョンずれと CI の多重化というコストを買うだけで
  重複も巨大ファイルも減らない。代わりに `.importlinter` から `ignore_imports` を
  全て消して契約をブロッキングにする（論理境界の厳格化）。

## 別途対処が必要な発見（本 spec の対象外）

以下はレビューで見つかったが Phase 1 では扱わない。Issue 化して追う。

- `longterm_backtest.py` のエントリーがリスクリーン当日 Close 約定である。
  `Backtester` は `#493` で `execution_lag=1`（翌バー約定）を既定にしてルック
  アヘッドを潰したが、この規律が新エンジンに適用されていない。docstring の
  「ルックアヘッド禁止」は screen の `as_of` の話のみを指しており、約定タイミングは
  対象外になっている。
- `build_conclusion` が `rescreen_freq` も `rules: HoldRules` も引数に取らないのに、
  結論文に「四半期スクリーン」「40 週線割れ」を文字列リテラルで埋めている。
  `--rescreen-freq weekly` で回すと結論文が嘘をつく。
- `longterm` の `_compute_metrics` は `metrics/core.py` の `compute_metrics` と並立
  しており、sharpe / profit_factor / DSR（過学習ガード `#372`）が一切かからない。
  キー名も `n_trades` と `num_trades` でずれている。
- リスクリーン日ごとに `screen_trend_candidates` が全銘柄を DB から読み直す
  （四半期 5 年 = 20 回 × 500 銘柄 = 1 万クエリ）。バックテストは同じ価格データを
  `price_map` に既に持っている。
- `.importlinter` のコメントにある「現状: 47 件」は実態と合っていない。
  `lint-imports` は exit 0 で通り、`ignore_imports` は longterm → screening の
  3 行のみである。
- `src/strategy/` と `src/features/` が空ディレクトリとして残っている。

---

## Phase 1 実施結果（2026-09-21 追記）

ブランチ `feature/execution-model`（develop から分岐）に 7 コミット。ユニット
テスト 2868 件パス、`lint-imports` exit 0、mypy 通過。23 ファイル / +1040 −154。
**未 push・PR 未作成。**

| PR | 状態 | 主なコミット |
|---|---|---|
| PR-1 ExecutionModel | 完了 | `b5af091` `75cf62d` `3a0a59f` `1483b28` |
| PR-2 CSV 保存ヘルパー | 完了 | `d3d369e` |
| PR-3 N+1 解消ほか | 完了（一部方針変更） | `3237ab1` |

### 設計から変更した判断

**1. CSV ヘルパーの置き場を `src/backtest/` から `src/utils/results_io.py` へ。**
5 箇所目の `trend_screener.py` は screening BC にあり、backtest 配下に置くと
screening からの参照が BC 独立性契約に違反するため。

**2. リバランス暦の共用は見送った。** `_get_rebalance_dates`（`pd.Grouper`・暦に
固定）と `_make_rescreen_dates`（`DateOffset`・開始日に固定）は重複ではなく別方針
であり、一本化するとどちらかの振る舞いが変わる。開始日に固定するのは長期
バックテストの正当な要件である。portfolio 側に quarterly の利用者も現状おらず、
既存テストは quarterly が `ValueError` になることを検証している。代わりに効率上の
実害 2 点（カレンダー線形走査、`_try_enter` の無条件に真となる再スライス）を潰した。

**3. longterm のエントリー株数は端株のまま維持した。** 一度 `max_affordable_qty`
（整数株）を当てたが、これはコスト以外の振る舞い変更で本 spec の範囲外。
`per_position / unit_buy_cost(price)` の逆算に改め、`buy_cost(shares, price) ==
per_position` が厳密に成立することを確認した。

### 未完了

- **実データの旧新差分レポートが未作成。** 本番 DB の読み取りが auto mode の
  classifier に "Production Reads" として拒否されたため。合成データでは受け入れ
  基準（悪化幅 ≒ 往復スリッページ率 × 売買回数）を満たすことを確認している
  （weekly: 理論上限 10.0pt に対し実測 8.09pt、monthly: 2.4pt に対し 1.94pt）。
- push と PR 作成。
