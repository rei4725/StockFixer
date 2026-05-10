# ADR-0003: `config/settings.py` と `config/trading_policy.py` の責務分離

- **日付**: 2025-06-01
- **ステータス**: Accepted
- **Roadmap ID**: R-218 / NF-503

## コンテキスト

設定値が 1 つのファイルに集まりすぎ、「環境によって変わる値」と「投資方針として決める値」が
混在していた。リスクプロファイルの切り替えやケリー基準のチューニングを行う際、
環境変数と事業上の意思決定が同一ファイルで管理されているため変更箇所が分かりにくかった。

## 検討した選択肢

| 選択肢 | 概要 |
|--------|------|
| 単一ファイル（settings.py に統合） | 全設定を 1 箇所に集約 |
| 2 ファイル分離（現行） | 環境設定と投資ポリシーを別ファイルに分割 |
| YAML/TOML 設定ファイル | 設定をコード外のファイルで管理 |

## 決定内容

`config/settings.py`（環境設定）と `config/trading_policy.py`（投資ポリシー）の 2 ファイルに分離する。

## 責務の境界

### `config/settings.py` が担うもの

- **環境依存の設定値**: `.env` ファイルや環境変数でオーバーライドされる値。
- `pydantic-settings` の `BaseSettings` を使い、型安全に環境変数を読み込む。
- 例: `MAX_DAILY_LOSS_RATE`, `BUY_THRESHOLD`, `PAPER_INITIAL_BALANCE`, `DISCORD_BOT_TOKEN` 等。
- 「インフラ・デプロイ・ランタイム」に関する設定。

### `config/trading_policy.py` が担うもの

- **投資判断・リスクプロファイルに関する設定値**: 「conservative / moderate / aggressive」の
  プロファイルに応じた最大ドローダウン許容率、ケリー上限、シャープ比閾値等。
- `RISK_PROFILE` 環境変数でプロファイルを選択できるが、各値は事業上の投資方針として定義される。
- 型・範囲バリデーションは `_strict_float()` で即時 `ValueError` を raise（`settings.py` の
  silent fallback とは意図的に異なる）。
- 例: `MAX_ACCEPTABLE_DRAWDOWN`, `KELLY_CAP`, `MIN_SHARPE_TO_TRADE` 等。

### 依存方向

```
settings.py  ──imports──→  trading_policy.py
```

`trading_policy.py` は `settings.py` を import してはならない（循環防止）。

## 理由

- **単一ファイルを採用しない理由**: 環境変数の値（デプロイ時に変わる）と投資方針の値（事業判断で変わる）が
  混在すると、変更の理由・影響範囲が不明確になる。レビュー時も diff が大きくなる。
- **YAML/TOML を採用しない理由**: コードとの型安全な連携が難しく、
  pydantic-settings によるバリデーションが失われる。また既存の import パターン
  （`from config.settings import MAX_DAILY_LOSS_RATE`）を維持できない。
- **2 ファイル分離を採用する理由**: 「どこを変えれば何が変わるか」が明確になる。
  投資方針の変更は `trading_policy.py` のみ、デプロイ設定の変更は `.env` + `settings.py` のみを見ればよい。

## 結果

**正の効果**:
- 環境設定と投資ポリシーの変更が独立して行える。
- `RISK_PROFILE=conservative` などプロファイル単位のプリセット切り替えが可能。
- `trading_policy.py` のバリデーションは範囲チェック付きの即時 `ValueError` で、
  不正な値での起動を防ぐ（fail-fast）。

**負の効果（トレードオフ）**:
- ファイルが 2 つになり、新しい設定値を追加する際にどちらに書くか判断が必要。
- `settings.py` が `trading_policy.py` を import するため、
  `trading_policy.py` 側から `settings.py` を参照することはできない（一方向依存の徹底が必要）。

## 関連

- `python/config/settings.py` — 環境依存設定
- `python/config/trading_policy.py` — 投資ポリシー設定
- `docs/ARCHITECTURE.md` — config 層の説明
