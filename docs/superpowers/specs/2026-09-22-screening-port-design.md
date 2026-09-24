# backtest → screening のポート化設計（Phase 3）

作成日: 2026-09-22
対象 BC: `src/backtest/`, `src/screening/`, `src/domain/`
先行 spec: `2026-09-21-execution-model-design.md`（Phase 1）、`2026-09-21-longterm-portfolio-types-design.md`（Phase 2）

## 背景

長期コホート・バックテストは screening BC の純粋関数と型を直接参照している。`.importlinter`
はこれを `ignore_imports` として両契約に明示登録しており、契約は通るが実質的に免除されている。

Phase 2 のパッケージ分割で、この免除は 3 行から 6 依存に増えた（各契約に 6 行、計 12 行）。

| 依存 | 参照するもの | 種類 |
|---|---|---|
| `longterm.engine → screening.trend_screener` | `screen_trend_candidates` | 関数 |
| `longterm.engine → screening.hold_engine` | `simulate_position` | 関数 |
| `longterm.engine → screening.types` | `TrendCandidate` | 型 |
| `longterm.portfolio → screening.types` | `PositionEvent` | 型 |
| `longterm.config → screening.types` | `HoldRules` | 型 |
| `longterm_backtest → screening.types` | `HoldRules`（ファサードの再輸出） | 型 |

Contract 2 の `ignore_imports` は現在この 6 依存だけである。したがって本 Phase を終えれば
**両契約の `ignore_imports` は空になる**。`.importlinter` のコメントにある「現状: 47 件」は
実態と合っていない古い記述であり、本 Phase で削除する。

### 関数と型で難易度が違う

関数 2 本はポートで素直に逆転できる。型 3 つはデータの形そのものが境界を跨ぐため、扱いが異なる。
特に `HoldRules` は backtest が**自分で構築している**（`LongtermBacktestConfig.rules` の
`field(default_factory=HoldRules)`）。戻り値の型ヒントとは性質が違い、構造的 Protocol では
代替できない。

## 検討した代替案

**案B（型）: backtest 側に構造的 Protocol を立てる。** screening の具象型を参照せず、
ダックタイピングの Protocol だけを backtest に置く。screening 側を一切触らずに済むが、
`HoldRules` は実クラスが無いと構築できないため成立しない。

**案B（関数）: `LongtermBacktestConfig` にポートを持たせる。** グローバル状態を持たずに済むが、
`start` や `top_n` と協力者オブジェクトが同じ frozen dataclass に同居することになり、
設定値の器としての意味が濁る。加えて既存 2 ポート（`backtest/data_port.py`,
`backtest/ports.py`）と流儀が割れる。

**案C: 関数を backtest 側へ複製する。** 却下。本番のスクリーニング実装が二重になる。

**採用: 型は `src/domain/` へ移し、関数は既存パターンのポートで逆転する。**

## 設計

### 型の移設

`HoldRules` / `PositionEvent` / `TrendCandidate` を `src/screening/types.py` から
`src/domain/types.py` へ移す。**定義はそのまま運ぶ** — フィールド名・型・既定値・docstring を
変更しない。`HoldRules.scale_out_multiples` の `field(default_factory=lambda: [2.0, 5.0])` も
原型どおりとする。

**再輸出はしない。** 利用者をすべて書き換える。型の正本を 1 箇所に保ち、「どちらから import
するのか」の迷いを残さないためである。

| 書き換え対象 | 件数 |
|---|---|
| `screening/{hold_engine,trend_screener,quality_gate,__init__}.py` | 4 |
| `orchestration/multibagger_job.py` | 1 |
| `backtest/longterm/{config,portfolio,engine}.py`, `backtest/longterm_backtest.py` | 4 |
| テスト | 8 |

`Position` / `MultibaggerCandidate` / `ValueCandidate` は `screening/types.py` に残す。
この 3 つは BC を跨がない。跨がない型を共有カーネルに積むと、カーネルが「なんでも置き場」に
堕ちる。

`src.domain` は layers 契約の `layers =` に列挙されていないため、どの BC から参照しても
契約違反にならない。共有カーネルとして正しい扱いである。

### ポートと結線

```
src/backtest/screening_port.py     BacktestScreeningPort（Protocol）+ set_ / get_
src/screening/backtest_adapter.py  BacktestScreeningAdapter（具象）
src/orchestration/port_wiring.py   wire_ports() に 1 行追加
run_longterm_backtest.py           wire_ports() を呼ぶ
```

```python
# src/backtest/screening_port.py
@runtime_checkable
class BacktestScreeningPort(Protocol):
    """backtest BC が screening BC に依存せず使うポート。"""

    def screen_trend_candidates(
        self, market: str, top_n: int, as_of: str
    ) -> list[TrendCandidate]: ...

    def simulate_position(
        self, prices: pd.DataFrame, entry_date: str, rules: HoldRules
    ) -> list[PositionEvent]: ...
```

`src/backtest/data_port.py` と同じ骨格にする。モジュールレベルの `_port` と
`set_backtest_screening_port()` / `get_backtest_screening_port()` を置き、未注入なら
`RuntimeError` を投げて `wire_ports()` を呼ぶよう促す。黙って `None` を返して後段で
`AttributeError` になる形は採らない。

アダプタは**提供側の BC**（`src/screening/`）に置く。`src/market_data/backtest_adapter.py`
と同じ対称性である。Protocol は構造的なので、アダプタが backtest を import する必要はない。
自 BC の関数は関数内 import で束ねる（既存アダプタと同じ書式）。

`engine.py` の直接呼び出しを `get_backtest_screening_port().screen_trend_candidates(...)` /
`.simulate_position(...)` に置き換える。ファサード `longterm_backtest.py` の `HoldRules`
再輸出は `src.domain.types` 由来に切り替わる。

### `run_longterm_backtest.py` の既存の穴

この CLI は現在 `wire_ports()` を呼んでいない。スケジューラは longterm を呼ばないため、
入口はこの 1 本だけである。ポート化に伴い `wire_ports()` の呼び出しを追加する。これは
`data_port` を使う経路に入れば今でも `RuntimeError` になる既存の穴でもあり、同時に塞ぐ。

引数パースとサービス呼び出しのみという `run_*.py` の規約は保つ（`wire_ports()` は合成であり
ビジネスロジックではない。他の `run_backtest*.py` も同じ形で呼んでいる）。

### `.importlinter` の後始末

両契約の `ignore_imports` を空にする。あわせて次の記述を整理する。

- Contract 2 の「現状: 47 件」という削減目標のコメントは実態と乖離している。実際の
  `ignore_imports` は本 Phase の対象 6 依存のみであり、本 Phase 完了で 0 件になる。
- 分類 [D]（backtest → screening）のロードマップ記述は解消済みとして削除する。

`unmatched_ignores = warn` は他契約にも効く設定であり、本 Phase では触らない。

## 振る舞い

**数値は 1 つも変わらない。純粋なリファクタリングである。**

型の移設は定義をそのまま運ぶだけで、関数のポート化は呼び出し経路に 1 段の委譲が挟まるだけである。
`screen_trend_candidates` と `simulate_position` の実装はどちらも変更しない。

## テスト

**既存の longterm テストが期待値を変えずに通ることを、正しさの証明とする。** Phase 2 の
PR-4 と同じ規律である。数値アサートの変更が必要になったら、それは移設か委譲にバグがある。

新規テスト:

| ファイル | 検証 |
|---|---|
| `tests/unit/backtest/test_screening_port.py` | 未注入で `RuntimeError`、メッセージが `wire_ports()` を示す。`set_` 後に `get_` が同じインスタンスを返す |
| `tests/unit/test_screening_backtest_adapter.py` | `isinstance(adapter, BacktestScreeningPort)` が真（`runtime_checkable`）。2 メソッドが実関数へ委譲する |
| `tests/unit/test_port_wiring.py` | `wire_ports()` 後に `get_backtest_screening_port()` が返る。冪等性 |

配置は既存の慣習に合わせる。`tests/unit/backtest/` はディレクトリが存在するのでその下に置き、
screening と orchestration のテストは `tests/unit/` 直下に平置きする（`test_hold_engine.py`,
`test_value_screener.py` と同じ）。`tests/unit/screening/` や `tests/unit/orchestration/` は
存在せず、本 Phase で新設もしない。`tests/unit/test_port_wiring.py` は新規である
（`wire_ports()` の既存テストは無い）。

### テスト密閉性

ポートはモジュールレベルのグローバル状態である。あるテストの注入が次のテストへ漏れると、
単体では通るのに実行順序次第で落ちる。`#516`（factory 化でテスト密閉性が壊れデプロイ失敗）と
同じ構造である。

既存の流儀に従って防ぐ。`tests/conftest.py` の autouse fixture `_wire_default_ports` が
毎テスト前に既定アダプタを注入し直しており、前のテストの注入は必ず上書きされる。この fixture に
`set_backtest_screening_port(BacktestScreeningAdapter())` を 1 行足す。個別に偽ポートを
注入するテストはそのまま上書きできる。

### 既存テストの patch

`tests/unit/backtest/longterm/` と `tests/unit/test_longterm_backtest.py` は現在
`patch.object(engine, "screen_trend_candidates")` というモジュール属性 patch を使っている。
これを偽ポートの注入に置き換える。

この置き換えには副次的な利得がある。モジュール属性 patch は関数の移動のたびに機械的に壊れ、
Phase 1 と Phase 2 で都度付け替えを強いられた。ポート注入になればその脆さが消える。

`load_price_map` と `fetch_benchmark_returns` は engine のモジュール属性のままなので、
そちらの patch は据え置く。

## PR 分割

型の移設だけで 6 依存中 4 つが消える。ここが自然な切れ目である。

| PR | 内容 | 消える依存 | version_impact |
|---|---|---|---|
| PR-7 | 3 型を `domain/types.py` へ移設し、全利用者を書き換える | 4 / 6 | patch |
| PR-8 | ポート導入・結線・`.importlinter` の `ignore_imports` を空にする | 2 / 6 | patch |

PR-7 は機械的な import 付け替えであり、レビューは「定義が 1 文字も変わっていないか」に集中できる。
PR-8 は構造変更のみで、型は既に片付いている。混ぜると、どちらが原因で壊れたのか事後に判別できない。

どちらも振る舞い不変のため `version_impact` は patch である。

## Phase 3 の対象外

- `unmatched_ignores = warn` の撤去（他契約にも影響するため別途判断）
- `screening/hold_engine.py` の逐次ステップ化（Phase 2 で案D として見送ったもの。ポート化が
  済んだ本 Phase の後に再検討できる）
- `portfolio/simulation.py` の生 dict（Phase 2 で対象外としたもの）
- 未知の `rescreen_freq` が quarterly として黙ってスケジュールされる一方、結論文は生文字列を
  ラベル表示する問題（Phase 2 で発見、CLI の `choices` が現状は防いでいる）
