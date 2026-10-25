# utils/ に混ざった業務の処理を domain/・BC へ移す 設計

- 日付: 2026-10-25
- 状態: 草案（人間の承認待ち）
- 関連: `docs/superpowers/specs/2026-09-22-db-ownership-decoupling-design.md` §3・§4、`python/.importlinter`

## 1. 意図と、なぜ今か

`python/.importlinter` の3つの契約（層・BC 独立・domain の純粋さ）は違反0件で守られている。
ただし BC 同士が直接 import できない代わりに、複数の BC が共有したい **売買の判断そのもの** が
`src/utils/` に置かれ、BC は `utils` を経由して業務の処理を分け合っている。

2026-09-22 の設計書 §4 の目標は「BC は domain と純粋 utils（logger 等）しか知らない」
「domain は型・ポート・ルール定数・例外」。今の `utils/` はこの目標とずれている。

実際に困った事故はまだ無い。人間の判断は **「今のうちにきれいにしておきたい」**。
ai-company が毎日 BC へ修正を積んでおり、`utils/` に業務の処理を置く前例が残ると増え続けるため、
増える前に片付け、戻らない仕組みを足す。

## 2. ゴール・ノンゴール・アンチゴール

**ゴール**
- 外とやり取りしない業務の処理（判断・表・暦）を `src/domain/` へ移す。
- 1つの BC しか使わない業務の処理を、その BC へ移す。
- `utils/` に新しい業務のモジュールが増えたら CI で止まる仕組みを足す。
- 移した後も、振る舞いは1ビットも変えない（同じ入力で同じ出力・同じ例外・同じログ）。

**ノンゴール**
- 外とやり取りする処理（yfinance でのセクターの照会、`optimal_params.json` の読み込み）を
  ポートと `infrastructure/` のアダプタに作り直すこと。BC の奥まで注入の道を通す必要があり
  （`get_optimal_params` の参照だけで 80 か所）、今回の「リスクを小さく」と合わない。
  後続の課題として §8 に残す。
- `utils/db/` の業務のテーブル（`experiment`・`strategy_promotions` など）。
- `utils/ml_base.py`（`BaseModel`）。joblib での保存と読み込みを持つ汎用の ML の道具で、
  売買の判断ではないため対象外（設計者の判断）。
- 600行の上限に張りついたファイルの分け直し。

**アンチゴール**（技術的に動いても失敗とみなす）
- 売買・バックテスト・予想の結果が1件でも変わること。
- `domain/` が `src.*`（`src.domain` 以外）や `config` を import すること。
- ai-company の作業中のブランチが、この移し替えだけを理由に CI で落ちること（§5 の転送で防ぐ）。
- テストの `mock.patch` の対象を書き換えないまま、patch が空振りして素通りする状態になること。

## 3. 制約

- 人間の手で操作するチェックアウト `C:/src/StockFixer` で作業する。ベースは `develop`。
  ブランチ名は `refactor/<対象>`。PR 本文の必須セクションは CLAUDE.md に従う。
- CI 一式（lint / mypy / pylint / import-linter / unit・cov≥80% / bandit / pip-audit）が
  PR ごとに緑であること。
- `.importlinter` の `ignore_imports` は0件のまま。
- 1ファイル600行の上限（NF-704）を守る。
- `domain/` は stdlib・pandas などの外部ライブラリだけを import してよい。
  `domain/` は `src.utils.logger` を import できず、CLAUDE.md は全モジュールに `get_logger` を求めるため、
  `domain/` へ移す処理はログを出さない純粋な関数・表にする。ログを出している部分は `utils/` 側に残す。

## 4. 移す先（決定事項）

| 今の場所 | 外とのやり取り | 移す先 | 備考 |
|---|---|---|---|
| `utils/regime_weights.py` | なし | `domain/regime_weights.py` | 表と `get_regime_sector_weight` をそのまま移す |
| `utils/jp_market_holidays.py` | なし | `domain/jp_market_holidays.py` | stdlib の `date` だけ使う |
| `utils/sector_constraints.py` の `filter_by_sector_cap` | なし | `domain/sector_cap.py` | `get_symbol_sector`（yfinance）は `utils/` に残す |
| `utils/signal_generator.py` の `generate_signal` の判断 | なし | `domain/signal_rules.py` の純粋関数 | 下の注を参照 |
| `utils/drift_thresholds.py` | 環境変数と DB（`utils.db.system_config.get_config_value`） | `reporting/drift_thresholds.py` | BC で使うのは reporting だけ。orchestration は上の層なので reporting を import してよい。reporting から `utils.db` を読むのは層として合法。**転送用の再輸出は置けない**（§5） |

**`SignalGenerator` の割り方**
- `domain/signal_rules.py` に、しきい値・出来高の窓・倍率をすべて引数で受け取る純粋関数を置く。
  `generate_signal` の今の本体（動的しきい値・RSI の拡張・出来高フィルター）をそのまま移す。
- `utils/signal_generator.py` の `SignalGenerator` クラスは残す。コンストラクタでの
  `optimal_params.json` の読み込みとログ、`config.settings` の
  `VOLUME_FILTER_WINDOW_DAYS`・`VOLUME_FILTER_MULTIPLIER` の読み込みは、このクラスに残す。
  `generate_signal` はその純粋関数を呼ぶだけの薄い入れ物になる。
  こうすると `utils/` に残るのは「データを取ってきて渡す」部分だけになる。
- 呼び出し元（`backtest/pipeline/runner.py`、`trading/signal_generator.py`）の書き換えは不要。

## 5. 移し方（決定事項）

- 移した元の場所には、**転送用の再輸出**を一時的に残す
  （例: `utils/regime_weights.py` が `from src.domain.regime_weights import *` 相当の明示の再輸出だけを持つ）。
  ai-company の作業中のブランチが古い import のままでも壊れないようにするため。
  - 例外: `drift_thresholds` は移す先が BC なので、`utils` に再輸出を置くと `utils → BC` の
    import になり層の契約が赤になる（`ignore_imports` も足さない）。これだけは転送を置かず一度に移す。
    使っているのは `orchestration/jobs/drift_check.py`・`reporting/kpi.py`・`reporting/monthly.py` の3か所だけなので、
    PR 3 を出す直前に、このファイルや import を触っている開いた PR が無いことを確かめる（設計者の判断）。
  - テストが外から触らぬ名前（例: `tests/unit/test_jp_market_holidays.py` の `_HOLIDAYS_BY_YEAR`）を
    import している場合は、テストの import を新しい場所へ書き換える。転送には公開の名前だけを載せる。
- `src/` の呼び出し元とテストの import は、移し替えと同じ PR で新しい場所へ書き換える。
  再輸出を経由している呼び出し元（例: `trading/signal_generator.py` の `utils.regime_weights` の再輸出）も含む。
- **ai-company の作業中のブランチの扱い**（設計者の判断）:
  - PR 1〜3 を出す直前に、移すファイルを触っている開いた PR を `gh pr list` と差分で確かめる。
    中身（例: 休場日の表）を書き換えている PR があれば、その PR のマージを先に待つ。
  - 転送が守るのは「古い場所を import しているだけ」のブランチに限る。中身が衝突した場合は、
    本体（`domain/` 側）に変更を入れ、転送のファイルには処理を戻さない。
  - 転送がある間に、新しいコードが古い場所を import するのは止めない（ブランチを落とさないため）。
    その分は PR 5 で書き換える。
- 転送用の再輸出は PR 5 で消す。PR 5 は PR 3 のマージから1週間後を目安に出し、
  その時点で古い場所を import している `develop` のコードと開いた PR は、PR 5 の中で書き換える
  （開いた PR は、そのブランチが CI で落ちたら ai-company が直す普段の流れに任せる）。
- `mock.patch` の文字列は、移した関数の **新しい場所** を指すように直す。
  今ある `src.utils.*` への patch 11 か所は、すべて `utils/` に残る外とのやり取りの部分
  （`sector_constraints.yf.Ticker`、`sector_constraints.get_symbol_sector`、
  `optimal_params_loader.OPTIMAL_PARAMS_PATH`）を指しているので、そのままでよい。
  移し替えの途中で新しく見つかった patch は同じ規則で直す。

## 6. 戻らない仕組み（決定事項）

- `utils/` の下（`utils/db/` などのサブパッケージも含む）に置いてよいモジュールの **許可リスト** を持ち、
  リストに無いモジュールが増えたら落ちるテストを足す。新しく足すときはリストに理由を書いて PR のレビューを通す。
  - 見るのはモジュール（ファイル）の単位まで。関数などの公開の名前までは固定しない
    （ai-company の普段の修正で頻繁に落ちる脆いテストになるため。設計者の判断）。
    既存のモジュールに業務の関数が足される道は、CLAUDE.md の一文とレビューで止める。
  - PR 4 の時点で残っている転送のモジュール（`regime_weights`・`jp_market_holidays` など）は、
    「一時的: PR 5 で消す」と理由を書いて許可リストに載せ、PR 5 でリストからも消す。
- CLAUDE.md の層の説明に「業務の判断・表・暦は `domain/`、1つの BC しか使わない処理はその BC。
  `utils/` は logger・DB・パス・リトライなどの道具だけ」と一文を足す。
- `.importlinter` の層の契約の説明の注釈（domain は「型とポートのみ」）を、
  §4 の目標（型・ポート・ルール定数・例外・純粋なルール関数）に合わせて直す。

## 7. PR の分け方（決定事項）と完了の確かめ方

1. `refactor/domain-pure-rules`: `regime_weights`・`jp_market_holidays`・`filter_by_sector_cap` を `domain/` へ。
2. `refactor/signal-rules-domain`: `SignalGenerator` の判断を `domain/signal_rules.py` へ。
3. `refactor/drift-thresholds-reporting`: `drift_thresholds` を `reporting/` へ。
4. `refactor/utils-allowlist`: 許可リストのテスト、CLAUDE.md・`.importlinter` の注釈の直し。
5. `refactor/remove-utils-shims`: 転送用の再輸出を消す（§5 の条件を満たしてから）。

各 PR の完了は次で確かめる。
- CI 一式が緑。
- 移したモジュールの既存のテストが、import の書き換え以外の変更なしで通る。
- `lint-imports` で3つの契約が緑で、`ignore_imports` が0件のまま。
- PR 2 では、`SignalGenerator` の既存のテスト（`tests/unit/test_signal_generator.py`）に加えて、
  純粋関数と入れ物のクラスが同じ入力に同じ `pd.Series` を返すことを確かめるテストを足す。

## 8. 後続の課題（今回はやらない）

- `get_symbol_sector`（yfinance）と `get_optimal_params`（ファイル）をポートにして
  `infrastructure/` が実装し、合成ルートから注入する。これが済むと 09-22 §4 の
  「BC は純粋 utils しか知らない」に届く。
- `utils/` に残る外とやり取りする道具（`ollama_client`・`github_api`・`alert_service` など）の置き場所の見直し。

## 9. 作り手に任せること

- `domain/` の新しいモジュールの中の関数名・引数の順・型注釈の細部（今の呼び出しと同じ振る舞いであること）。
- 許可リストのテストの置き場所と書き方。
- テストの import の書き換えを、どの PR にどこまで含めるかの細かい割り振り。
