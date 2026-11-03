# バックテストによる暫定昇格 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 学習に締め日を足し、取っておいた直近20営業日で「今のコードで学んだ候補」と「今の production」を本番と同じ予想の道で比べ、勝てば自動で暫定昇格、5営業日後に1回見張って必要なら戻す。

**Architecture:** 予想の BC（`src/prediction/backtest_promotion/`）に、判定の純粋関数（`judge.py`）・取っておいた期間の評価（`holdout.py`）・状態とファイルの入れ替え（`state.py`）を置く。オーケストレーション層のジョブ（`src/orchestration/jobs/backtest_promotion_job.py`）がそれをつなぎ、JSON と Discord に出す。本番の予想の関数からアンサンブルの芯を切り出し、本番とバックテストの両方がそれを呼ぶ。

**Tech Stack:** Python 3 / pandas / XGBoost・LightGBM（既存の `ModelManager`）/ `filelock` / APScheduler（既存の `SCHEDULE_CONFIG`）/ pytest

**Spec:** `docs/superpowers/specs/2026-11-03-backtest-provisional-promotion-design.md`

## Global Constraints

- 取っておく期間: 直近 **20** 営業日（`stock_features` の `y` が埋まった日付の、新しい方から20個）
- 勝ち: `hit_rate_minus_majority` 候補 > production、`mae` 候補 < production、`rank_corr` 候補 > production かつ 候補 > 0、日ごとの `hit_rate_minus_majority` の勝ち越しが過半数（> 50%）、各モデルの `n_baseline` ≥ 20 × 対象銘柄数 × **0.8**
- 向きだけの的中率は記録のみ、判定に使わない
- 見張り: 入れ替え後、採点済みの日が **5** 営業日たまったら1回だけ判定。条件は「新 production の `hit_rate_minus_majority` がバックテストの候補の値より **0.03** 以上低い」「`rank_corr` ≤ 0」「`predicted_up_rate` > **0.9** または < **0.1**」のどれか。かつ戻す先（旧 production）のバックテストの `hit_rate_minus_majority` と `rank_corr` が両方とも新 production の5日の値を上回るときだけ戻す
- 戻した候補と同じコードの版からは再び入れ替えない
- 道の一致の検査: 最新の日について、最大 **50** 銘柄で本番の `predict_with_unified_model` とバックテストの道の `diff_ratio` を比べ、差の絶対値がすべて **1e-4** 以下（本番が None の銘柄は除く）。外れたら入れ替えない
- 入れ替えるのはバックテストで測った候補そのもの（XGBoost と LightGBM のみ。Transformer は対象外）
- 失敗（例外・データが空）は「負け」にせず失敗として JSON と Discord に出す
- 本物の口座（`live_trading`）・売買のルール・週次の challenger の流れの判定は変えない
- 層の規則: `src/prediction/` は `orchestration/`・`reporting/` を import しない。`run_*.py` は引数の解釈と呼び出しだけ
- 1日1回まで。同じ版での再実行はしない（`--force` を除く）
- コミットは Conventional Commits、日本語の件名。PR のベースは `develop`。VERSION は minor を上げる（2.35.0 → 2.36.0、PR 本文の必須セクションも書く）

## Review Focus

1. **取っておいた期間に、候補か production の一方しか予想が出ない銘柄・日がある**（特徴量が古い、モデルが落ちたなど）。物差しは両方の予想がそろった (market, symbol, date) の行だけで出すべきで、片方だけの行で比べてはならない → Task 4 のテスト `test_metrics_use_only_rows_both_models_predicted`
2. **入れ替えの途中で落ちる**（写しは取れたが2本目のコピーで失敗）。本番のファイルが候補と旧の混ざった状態のまま、状態のファイルが「入れ替え済み」にならないこと → Task 5 のテスト `test_swap_failure_on_second_copy_restores_and_raises`
3. **週次学習（火 01:00）と夜間のジョブ（22:00）が重なる**。一方がモデルのファイルを触っている間、もう一方は待つべき → Task 5 のロックと Task 6 のテスト `test_weekly_training_holds_unified_model_lock`
4. **production のファイルが無い・候補の学習が例外で落ちる・`stock_features` が空**。どれも「負け」と記録してはならず、版を「評価済み」にもしない（次の夜にやり直す） → Task 6 のテスト `test_failure_is_not_recorded_as_loss_and_version_not_marked`
5. **見張りの5日の間に週次の昇格が起きる**。戻す先が分からなくなるので見張りをやめる → Task 6 のテスト `test_weekly_promotion_clears_watch`

---

## File Structure

| ファイル | 役目 |
|---|---|
| `python/src/prediction/unified_model_pipeline.py`（変更） | `train_unified_model` に締め日と、保存した除く列・特徴量選びの保存を切る引数を足す |
| `python/src/prediction/predict_unified.py`（変更） | アンサンブルの芯 `ensemble_diff_ratio` を切り出す。`predict_with_unified_model` はそれを呼ぶ |
| `python/src/prediction/backtest_promotion/__init__.py`（新規） | 公開する名前の re-export |
| `python/src/prediction/backtest_promotion/judge.py`（新規） | 定数と、勝ち・戻しの判定の純粋関数 |
| `python/src/prediction/backtest_promotion/holdout.py`（新規） | 締め日の決定、取っておいた期間の予想と物差し、道の一致の検査 |
| `python/src/prediction/backtest_promotion/state.py`（新規） | 状態のファイル、コードの版、ロック、候補の入れ替えと戻し |
| `python/src/prediction/types.py`（変更） | 結果の dataclass |
| `python/src/prediction/shadow_evaluation.py`（変更） | `promote_unified_challenger` の写しとコピーを `state.py` の共通の関数に寄せる |
| `python/src/orchestration/jobs/backtest_promotion_job.py`（新規） | 夜間のジョブ（評価→入れ替え、見張り→戻し、JSON、Discord） |
| `python/src/orchestration/jobs/weekly_training.py`（変更） | ロックを取る、昇格したら見張りをやめる |
| `python/src/orchestration/scheduler.py`（変更） | ジョブを re-export |
| `python/run_scheduler.py`（変更） | `SCHEDULE_CONFIG` に `nightly_backtest_promotion` を足す |
| `python/run_backtest_promotion.py`（新規） | 手で動かす入口 |
| `python/src/reporting/discord/notifications_model.py`（変更） | 通知3種 |
| `python/tests/unit/prediction/backtest_promotion/test_*.py`（新規） | 単体テスト |

---

### Task 1: 締め日つきの学習（漏れを防ぐ）

**Files:**
- Modify: `python/src/prediction/unified_model_pipeline.py`（`prepare_unified_features`、`train_unified_model`）
- Test: `python/tests/unit/prediction/backtest_promotion/test_cutoff_training.py`

**Interfaces:**
- Produces:
  - `drop_rows_reaching_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame` — 銘柄（market, symbol）ごとに日付順で、**次の行の日付が `cutoff` 未満**の行だけを残す（`y` は次の行の終値を使う。`market_data/technical.py` の `target = Close.shift(-1)`）。最後の行（次の行が無い）も落とす
  - `prepare_unified_features(df, apply_saved_exclusions: bool = True)` — False のとき `_apply_unified_feature_exclusions` を呼ばない
  - `train_unified_model(..., train_end_date: Optional[pd.Timestamp] = None, apply_saved_exclusions: bool = True, save_selection: bool = True)` — `train_end_date` を渡したときは `load_all_stock_data` の直後に `drop_rows_reaching_cutoff` をかける。`horizon != 1` と一緒に渡されたら `ValueError`。`save_selection=False` のときは `_compute_and_save_unified_feature_selection` を呼ばない。省略時の振る舞いは今と同じ

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_drop_rows_reaching_cutoff_keeps_only_rows_whose_next_day_is_before_cutoff():
    # us/A: 10-01,10-02,10-03,10-06 / jp/B: 10-01,10-03 、cutoff=10-03
    out = drop_rows_reaching_cutoff(df, pd.Timestamp("2026-10-03"))
    assert set(map(tuple, out[["market", "symbol", "date"]].astype(str).values)) == {
        ("us", "A", "2026-10-01"),
    }  # us/A 10-02 は次が 10-03 なので落ちる、jp/B 10-01 も次が 10-03 なので落ちる

def test_cutoff_training_ignores_marked_targets_after_cutoff(monkeypatch, tmp_path):
    # 締め日以降の行の y を 1e6、特徴量 f にもその目印を入れたデータで学習する
    # 1) 締め日つきで学習したモデルと 2) 締め日以降の行を最初から除いたデータで学習したモデルが、
    #    同じ入力に同じ予想を返す（random_state は固定）
    # 3) 分け方に渡った最大の日付 < 締め日
    # 4) save_selection=False のとき save_feature_selection が呼ばれない
    ...
    assert np.allclose(pred_with_cutoff, pred_pre_trimmed)
    assert max_split_date < cutoff
    assert save_selection_calls == []

def test_cutoff_with_multi_horizon_raises():
    with pytest.raises(ValueError):
        train_unified_model(horizon=3, train_end_date=pd.Timestamp("2026-10-01"))
```

テスト2は `load_all_stock_data` を monkeypatch して小さい表（2銘柄 × 60日、特徴量 3列）を返す。`ModelManager` は実物の XGBoost を `n_estimators=5` で使う。

- [ ] **Step 2: 落ちることを確かめる**

Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_cutoff_training.py -v`
Expected: FAIL（`drop_rows_reaching_cutoff` が無い）

- [ ] **Step 3: 実装する**（Interfaces のとおり。`stock_features` の特徴量の作りに未来の値を使うものが無いかを `market_data/technical.py` で確かめ、見つけたらこの Task の PR 説明に書く。直すのはこの計画の外）

- [ ] **Step 4: 通ることを確かめる**

Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_cutoff_training.py tests/unit/ -k "unified_model_pipeline or cutoff" -v`
Expected: PASS（既存の unified_model_pipeline のテストも通る）

- [ ] **Step 5: コミット**

```bash
git add python/src/prediction/unified_model_pipeline.py python/tests/unit/prediction/backtest_promotion/
git commit -m "feat: 統合モデルの学習に締め日を足す"
```

---

### Task 2: アンサンブルの芯を切り出す

**Files:**
- Modify: `python/src/prediction/predict_unified.py:277-510`
- Modify: `python/src/prediction/types.py`
- Test: `python/tests/unit/prediction/backtest_promotion/test_ensemble_core.py`

**Interfaces:**
- Produces:
  - `EnsembleOutcome`（`types.py` の dataclass）: `diff_ratio_raw: float`（クリップ前）, `diff_ratio: float`（クリップ後）, `clip_max_ratio: Optional[float]`, `model_count: int`, `zero_filled: list[str]`
  - `ensemble_diff_ratio(latest_X: pd.DataFrame, current_price: float, market: str, symbol: str, horizon: int, models: list[tuple[str, Any]], weights_fn: Callable[[str, str, list[str]], list[float]] = load_model_weights) -> Optional[EnsembleOutcome]` — `latest_X` は1行。`models` は（重みを引く名前, モデルの実物）。今の関数の「0埋め・列そろえ → predict → 重み → ATR クリップ」をそのまま移す。`record_zero_fill`・`record_clip`・ログは**呼ばない**（呼ぶのは本番の側）。全モデルが例外なら今と同じく `RuntimeError`、モデルが1つも無ければ None
  - `predict_with_unified_model` は `get_cached_model` で `models` を組み、`ensemble_diff_ratio` を呼び、`record_zero_fill`・`record_clip`・ログを今と同じ順で出す。戻り値・例外・推論サービスへの委譲は変えない

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_ensemble_core_matches_live_function(monkeypatch):
    # 2つの偽モデル（予想 0.02 と 0.04 を返す）、重み [0.25, 0.75]、atr_lag1 と価格はクリップしない値
    out = ensemble_diff_ratio(row, 100.0, "us", "A", 1, [("X", m1), ("L", m2)], weights_fn=lambda *a: [0.25, 0.75])
    assert out.diff_ratio == pytest.approx(0.035)
    assert out.model_count == 2
    live = predict_with_unified_model("us", "A")  # 同じ偽モデル・行・価格を monkeypatch
    assert live.diff_ratio == pytest.approx(out.diff_ratio, abs=1e-12)

def test_ensemble_core_does_not_record_run_stats(monkeypatch):
    # record_zero_fill / record_clip が呼ばれないこと
    ...

def test_ensemble_core_uses_weight_names_not_model_objects():
    # weights_fn に渡る名前が models の第1要素であること
    ...
```

- [ ] **Step 2: 落ちることを確かめる** — Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_ensemble_core.py -v` / Expected: FAIL（`ensemble_diff_ratio` が無い）
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通ることを確かめる** — Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_ensemble_core.py tests/unit/ -k "predict_unified" -v` と、別に直列で `python -m pytest tests/unit/test_predict_unified_lightgbm_alignment.py -v` / Expected: すべて PASS
- [ ] **Step 5: コミット** — `git commit -m "refactor: 統合モデルのアンサンブルの芯を切り出す"`

---

### Task 3: 勝ちと戻しの判定（純粋関数）

**Files:**
- Create: `python/src/prediction/backtest_promotion/__init__.py`, `python/src/prediction/backtest_promotion/judge.py`
- Modify: `python/src/prediction/types.py`
- Test: `python/tests/unit/prediction/backtest_promotion/test_judge.py`

**Interfaces:**
- Produces（`judge.py`）:
  - 定数: `HOLDOUT_TRADING_DAYS = 20`, `MIN_COVERAGE = 0.8`, `WATCH_TRADING_DAYS = 5`, `ROLLBACK_HIT_MARGIN = 0.03`, `UP_RATE_LOW = 0.1`, `UP_RATE_HIGH = 0.9`, `PATH_CHECK_SAMPLE = 50`, `PATH_CHECK_TOL = 1e-4`
  - `WinDecision`（`types.py`）: `won: bool`, `reasons: list[str]`（落ちた条件ごとに1つ、日本語）
  - `judge_win(candidate: dict, production: dict, daily_diff: list[float], expected_rows: int) -> WinDecision` — `candidate`・`production` は `baseline_metrics()` の dict。`daily_diff` は日ごとの（候補 − production）の `hit_rate_minus_majority`。`expected_rows` = 20 × 対象銘柄数。値が None の物差しはその条件の負けにする
  - `RollbackDecision`（`types.py`）: `rollback: bool`, `triggered: list[str]`, `old_is_better: bool`
  - `judge_rollback(live: dict, backtest_candidate: dict, backtest_old_production: dict) -> RollbackDecision` — Global Constraints の見張りの条件。`rollback = bool(triggered) and old_is_better`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_always_up_predictions_cannot_win():
    rows = pd.DataFrame({"predicted_ratio": [0.06] * 400, "actual_ratio": np.r_[np.full(210, 0.01), np.full(190, -0.01)]})
    m = baseline_metrics(rows)
    assert abs(m["hit_rate_minus_majority"]) < 1e-9
    assert judge_win(m, m, [0.0] * 20, 400).won is False

def test_win_requires_all_three_metrics():
    good = {"hit_rate_minus_majority": 0.03, "mae": 0.015, "rank_corr": 0.05, "n_baseline": 1000}
    prod = {"hit_rate_minus_majority": -0.01, "mae": 0.06, "rank_corr": 0.01, "n_baseline": 1000}
    assert judge_win(good, prod, [0.01] * 11 + [-0.01] * 9, 1000).won is True
    assert judge_win({**good, "mae": 0.07}, prod, [0.01] * 20, 1000).won is False
    assert judge_win({**good, "rank_corr": -0.01}, {**prod, "rank_corr": -0.05}, [0.01] * 20, 1000).won is False

def test_win_requires_daily_majority_and_coverage():
    assert judge_win(good, prod, [0.01] * 10 + [-0.01] * 10, 1000).won is False   # ちょうど半分は負け
    assert judge_win({**good, "n_baseline": 799}, prod, [0.01] * 20, 1000).won is False

def test_rollback_only_when_triggered_and_old_is_better():
    live = {"hit_rate_minus_majority": -0.02, "rank_corr": 0.01, "predicted_up_rate": 0.5}
    bt_cand = {"hit_rate_minus_majority": 0.02}
    assert judge_rollback(live, bt_cand, {"hit_rate_minus_majority": 0.0, "rank_corr": 0.02}).rollback is True
    d = judge_rollback(live, bt_cand, {"hit_rate_minus_majority": -0.05, "rank_corr": 0.02})
    assert d.triggered and d.rollback is False and d.old_is_better is False
    assert judge_rollback({**live, "predicted_up_rate": 0.95, "hit_rate_minus_majority": 0.02},
                          bt_cand, {"hit_rate_minus_majority": 0.03, "rank_corr": 0.05}).rollback is True
    assert judge_rollback({"hit_rate_minus_majority": 0.0, "rank_corr": 0.01, "predicted_up_rate": 0.5},
                          bt_cand, {"hit_rate_minus_majority": 0.05, "rank_corr": 0.05}).triggered == []
```

- [ ] **Step 2: 落ちることを確かめる** — Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_judge.py -v` / Expected: FAIL（import できない）
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通ることを確かめる** — 同じコマンドで PASS
- [ ] **Step 5: コミット** — `git commit -m "feat: バックテストの勝ちと戻しの判定を足す"`

---

### Task 4: 取っておいた期間の評価と道の一致の検査

**Files:**
- Create: `python/src/prediction/backtest_promotion/holdout.py`
- Modify: `python/src/prediction/types.py`
- Test: `python/tests/unit/prediction/backtest_promotion/test_holdout.py`

**Interfaces:**
- Consumes: `ensemble_diff_ratio`（Task 2）, `judge_win` と定数（Task 3）, `baseline_metrics`, `load_stock_features`（`src/utils/db`）, `load_raw_ohlcv`（終値）, `load_all_raw_ohlcv_symbols`
- Produces:
  - `holdout_dates(feature_dates: pd.Series, n: int = HOLDOUT_TRADING_DAYS) -> list[pd.Timestamp]` — `y` が埋まった行の日付の重複なしの新しい方から n 個を昇順で。締め日 = その先頭
  - `HoldoutEvaluation`（`types.py`）: `cutoff: str`, `dates: list[str]`, `n_symbols: int`, `candidate: dict`, `production: dict`, `daily: list[dict]`（`date`, `candidate`, `production`, `diff` の `hit_rate_minus_majority`）, `path_check: dict`（`compared`, `max_abs_diff`, `passed`）, `recorded_reference: dict`（`prediction_accuracy` の production との突き合わせ。参考のみ）, `decision: WinDecision`
  - `evaluate_holdout(cutoff_dates: list[pd.Timestamp], candidate_models: list[tuple[str, Any]], production_models: list[tuple[str, Any]], live_predict: Callable[[str, str], Optional[PredictionResult]]) -> HoldoutEvaluation`
    - 銘柄ごとに `load_stock_features` で1回読み、各取っておいた日の行を `latest_X` として `ensemble_diff_ratio` を候補・production で呼ぶ。重みの名前はどちらも `["UnifiedStockXGBoost", "UnifiedStockLightGBM"]`（入れ替えた後に候補が本番で使う重みと同じになる）。価格は `load_raw_ohlcv` のその日の `Close`。`actual_ratio` はその行の `y`
    - 物差しは**両方の予想がそろった行だけ**で出す
    - 道の一致の検査: 最新の日（`stock_features` の最後の行）について、ランダムでない先頭 `PATH_CHECK_SAMPLE` 銘柄（market, symbol の順）で `live_predict` と同じ行のバックテストの道の production の `diff_ratio` を比べる。`passed = compared > 0 and max_abs_diff <= PATH_CHECK_TOL`
    - `decision` は `judge_win`。`path_check.passed` が False なら `decision.won=False`、`reasons` に「道の一致の検査に落ちた」を足す

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_holdout_dates_are_latest_n_dates_with_target():
    # 25日分、最後の日は y が NaN → 最後から2番目までの20日
    assert holdout_dates(dates_with_target, 20) == expected_20

def test_metrics_use_only_rows_both_models_predicted(monkeypatch):
    # 候補が1銘柄の1日でモデル無し（None）を返す → その行は production の物差しからも外れる
    ev = evaluate_holdout(...)
    assert ev.candidate["n_baseline"] == ev.production["n_baseline"] == total_rows - 1

def test_candidate_uses_production_weight_names(monkeypatch):
    # weights_fn に渡る名前が候補でも ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
    ...

def test_path_check_failure_blocks_win(monkeypatch):
    # live_predict が 0.001 ずれた diff_ratio を返す → passed False、decision.won False
    ...
```

- [ ] **Step 2: 落ちることを確かめる** — Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_holdout.py -v` / Expected: FAIL
- [ ] **Step 3: 実装する**（`recorded_reference` は `load_prediction_accuracy` から model_name=production・horizon=1 の同じ (market, symbol, date) を引き、向きの一致率と `predicted_ratio` の差の中央値を書くだけ）
- [ ] **Step 4: 通ることを確かめる** — 同じコマンドで PASS
- [ ] **Step 5: コミット** — `git commit -m "feat: 取っておいた期間で候補と production を比べる"`

---

### Task 5: 状態・コードの版・ロック・入れ替えと戻し

**Files:**
- Create: `python/src/prediction/backtest_promotion/state.py`
- Modify: `python/src/prediction/shadow_evaluation.py:438-510`（`promote_unified_challenger` の写しとコピーを `copy_with_backup` に寄せる。戻り値と振る舞いは変えない）
- Test: `python/tests/unit/prediction/backtest_promotion/test_state.py`

**Interfaces:**
- Produces:
  - 定数: `CANDIDATE_NAMES = ["UnifiedStockXGBoost_candidate", "UnifiedStockLightGBM_candidate"]`, `SWAP_PRODUCTION_NAMES = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]`
  - `results_dir() -> Path` = `Path(get_results_dir()) / "backtest_promotion"`。状態は `results_dir() / "state.json"`
  - `BacktestPromotionState`（`types.py`）: `last_evaluated_version: Optional[str]`, `rolled_back_versions: list[str]`, `watch: Optional[dict]`（`version`, `swapped_at`（JST の ISO）, `swap_date`（YYYY-MM-DD）, `backups: list[str]`, `backtest_candidate: dict`, `backtest_old_production: dict`）
  - `load_state() -> BacktestPromotionState` / `save_state(state) -> None`（一時ファイルに書いて置き換える）
  - `prediction_code_version() -> str` — `python/src/prediction/**/*.py` と `python/src/market_data/**/*.py` を相対パスの順に読み、パスと中身の sha256 の先頭12桁。`__pycache__` は除く
  - `unified_model_lock(timeout_s: float = 3 * 3600) -> FileLock` — `models/unified/.unified_models.lock`
  - `copy_with_backup(pairs: list[tuple[str, str]], stamp: str) -> list[str]` — (src, dst) の組。先に全部の dst を `<名前>_<stamp>.joblib` に写し、その後コピーする。コピーの途中で落ちたら、写しから全部の dst を戻して例外を投げ直す
  - `swap_in_candidate() -> list[str]`（写しのパス）/ `restore_backups(backups: list[str]) -> None`（戻す前に今のファイルも写す）
  - `clear_watch(reason: str) -> None` — 見張りがあれば消し、理由をログに出す

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_swap_failure_on_second_copy_restores_and_raises(tmp_path, monkeypatch):
    # 2本目の shutil.copy2 で OSError → 例外、両方の本番のファイルが元の中身のまま
    ...

def test_swap_then_restore_round_trip(tmp_path, monkeypatch):
    backups = swap_in_candidate()
    assert read(prod_xgb) == read(cand_xgb)
    restore_backups(backups)
    assert read(prod_xgb) == original_xgb

def test_code_version_changes_only_with_prediction_or_market_data_code(tmp_path, monkeypatch):
    ...

def test_state_round_trip_and_atomic_write(tmp_path, monkeypatch):
    ...

def test_promote_unified_challenger_unchanged_behavior(tmp_path, monkeypatch):
    # 既存の shadow_evaluation のテストがそのまま通ること（返り値 promoted/skipped/backups/dry_run）
    ...
```

- [ ] **Step 2: 落ちることを確かめる** — Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_state.py -v` / Expected: FAIL
- [ ] **Step 3: 実装する**
- [ ] **Step 4: 通ることを確かめる** — Run: `cd python && python -m pytest tests/unit/prediction/backtest_promotion/test_state.py tests/unit/ -k "shadow_evaluation or promote" -v` / Expected: PASS
- [ ] **Step 5: コミット** — `git commit -m "feat: 暫定昇格の状態と入れ替え・戻しを足す"`

---

### Task 6: 夜間のジョブ・通知・入口・週次との排他

**Files:**
- Create: `python/src/orchestration/jobs/backtest_promotion_job.py`, `python/run_backtest_promotion.py`
- Modify: `python/src/orchestration/jobs/weekly_training.py`, `python/src/orchestration/scheduler.py`, `python/run_scheduler.py`（`SCHEDULE_CONFIG`）, `python/src/reporting/discord/notifications_model.py`, `VERSION`
- Test: `python/tests/unit/orchestration/test_backtest_promotion_job.py`

**Interfaces:**
- Consumes: Task 1〜5 のすべて
- Produces:
  - `run_backtest_promotion(force: bool = False, dry_run: bool = False) -> dict` — 次の順で動く。全体を `unified_model_lock()` の中で動かす
    1. **見張り**: `state.watch` があれば、`prediction_accuracy`（model_name=production, horizon=1, `predicted_at` の日付 > `swap_date`, `actual_ratio` あり）の採点済みの日を数える。5日以上なら `baseline_metrics` → `judge_rollback`。戻すなら `restore_backups`・`rolled_back_versions` に版を足す。どちらでも見張りを消し、結果を通知する。5日未満なら何もしない
    2. **評価**: `version = prediction_code_version()`。`version == last_evaluated_version` または `version in rolled_back_versions` で `force` が無ければ終える。見張り中も評価しない
    3. `holdout_dates` → 締め日。`train_unified_model(model_type, model_name=<候補名>, train_end_date=cutoff, apply_saved_exclusions=False, save_selection=False)` を XGBoost と LightGBM で
    4. `evaluate_holdout(...)`。`live_predict` には `predict_with_unified_model` を渡す
    5. 勝ちかつ `dry_run` でなければ `swap_in_candidate()`、`state.watch` を作る
    6. `last_evaluated_version = version` を保存（**3〜5 で例外が出たときは保存しない**）
    7. `results_dir() / f"{YYYYMMDD_HHMMSS}.json"` に、版・締め日・期間・件数・物差し・日ごとの差・道の一致の検査・参考の突き合わせ・判定と理由・入れ替えたか・写し・`dry_run`・失敗ならその例外の文字列を書く
    8. Discord に1通（勝ち／負け／比べが成り立たない／失敗）、入れ替えたら1通、戻したら1通
  - `notifications_model.py`: `send_backtest_promotion_result(result: dict) -> bool`, `send_backtest_promotion_swap(result: dict) -> bool`, `send_backtest_promotion_rollback(result: dict) -> bool` — 今の `send_shadow_evaluation_notification` と同じ作り（spec の定数・Webhook）
  - `run_scheduler.py` の `SCHEDULE_CONFIG["nightly_backtest_promotion"]`: `trigger="cron"`, `period="daily"`, `day_of_week="mon-sun"`, `hour=22`, `minute=0`, `recovery_delay_minutes=60`, `max_executions_per_period=1`, `description="毎日 22:00 - 予想のコードが変わったらバックテストで候補を評価し暫定昇格"`
  - `run_backtest_promotion.py`: `--force`, `--dry-run` を受けて `run_backtest_promotion` を呼び、結果の JSON のパスを出す
  - `weekly_training.py`: `run_weekly_training` の本体を `unified_model_lock()` の中で動かす。`promoted = True` になったら `clear_watch("週次の昇格")`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_same_version_is_not_reevaluated(monkeypatch, tmp_path): ...
def test_rolled_back_version_is_not_reevaluated_without_force(monkeypatch, tmp_path): ...
def test_win_swaps_and_starts_watch(monkeypatch, tmp_path): ...
def test_dry_run_never_swaps(monkeypatch, tmp_path): ...
def test_failure_is_not_recorded_as_loss_and_version_not_marked(monkeypatch, tmp_path):
    # train_unified_model が RuntimeError → JSON の outcome が "failed"、last_evaluated_version は前のまま、
    # send_backtest_promotion_result に outcome="failed" が渡る
    ...
def test_watch_waits_until_five_scored_days(monkeypatch, tmp_path): ...
def test_watch_rolls_back_and_records_version(monkeypatch, tmp_path): ...
def test_watch_triggered_but_old_worse_keeps_model_and_notifies(monkeypatch, tmp_path): ...
def test_weekly_training_holds_unified_model_lock(monkeypatch): ...
def test_weekly_promotion_clears_watch(monkeypatch, tmp_path): ...
def test_schedule_config_has_nightly_backtest_promotion():
    from run_scheduler import SCHEDULE_CONFIG
    c = SCHEDULE_CONFIG["nightly_backtest_promotion"]
    assert (c["day_of_week"], c["hour"], c["minute"]) == ("mon-sun", 22, 0)
```

- [ ] **Step 2: 落ちることを確かめる** — Run: `cd python && python -m pytest tests/unit/orchestration/test_backtest_promotion_job.py -v` / Expected: FAIL
- [ ] **Step 3: 実装する**（`VERSION` を 2.36.0 に）
- [ ] **Step 4: 通ることを確かめる** — Run: `cd python && .\check-ci.ps1`（Windows）/ Expected: lint・mypy・import-linter・単体テスト（cov ≥ 80%）すべて緑
- [ ] **Step 5: コミット** — `git commit -m "feat: バックテストによる暫定昇格の夜間ジョブを足す"`

---

### Task 7: 本番のデータで確かめる（デプロイの後）

**Files:** なし（確かめのみ）

- [ ] **Step 1: dry_run で1回動かす** — Run: `docker-compose exec stockfixer python run_backtest_promotion.py --dry-run --force` / Expected: `results/backtest_promotion/*.json` ができ、`path_check.passed` が true、`candidate`・`production` の `n_baseline` が 20 × 銘柄数の 80% 以上、`dry_run: true`、モデルのファイルの mtime が変わっていない
- [ ] **Step 2: JSON の物差しを手で確かめる** — production の行の `predicted_up_rate` が 0.9 を超えていること（Issue 1309 の病気が見えていること）を確かめ、PR か Issue 1309 にコメントで書く
- [ ] **Step 3: 本番の回を待つ** — 22:00 の回で入れ替えが起きたら Discord の通知と `state.json` の `watch` を確かめる。5営業日後の見張りの判定も確かめる
