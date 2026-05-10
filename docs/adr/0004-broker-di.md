# ADR-0004: `BrokerBase` 抽象化による依存性注入（DI）採用

- **日付**: 2024-06-01
- **ステータス**: Accepted
- **Roadmap ID**: NF-503

## コンテキスト

StockFixer は当初 SBI 証券 API との直接連携を前提に設計されていたが、
証券会社の変更・追加（Kabu Station、将来的な楽天証券等）や
ペーパートレード（仮想売買）モードへの切り替えが必要になった。
services 層が特定の証券会社実装に直接依存すると、証券会社を変更するたびに
ビジネスロジック側のコードを修正しなければならない。

## 検討した選択肢

| 選択肢 | 概要 |
|--------|------|
| 具体クラスを直接参照 | services 層が `KabuBroker` や `PaperBroker` を直接 import |
| ファクトリー関数 | 文字列で Broker を選択するファクトリーを用意 |
| 抽象基底クラス（ABC）+ DI | `BrokerBase` ABC を定義し、呼び出し元から注入する |

## 決定内容

`BrokerBase` 抽象基底クラス（ABC）を `src/trading/brokers/base.py` に定義し、
services 層は `BrokerBase` のみを参照する依存性注入パターンを採用する。

## 責務の構造

```
src/trading/brokers/
├── base.py          ← BrokerBase ABC（OrderSide / OrderType 列挙型も定義）
├── paper/           ← PaperBroker（DuckDB バックエンド仮想売買）
└── kabu/            ← KabuBroker（Kabu Station® API 実連携）
```

`BrokerBase` が定義する抽象メソッド:
- `get_token()` — 認証トークン取得
- `send_order()` — 注文発注
- `cancel_order()` — 注文キャンセル
- `get_balance()` — 残高照会
- `get_orders()` — 注文一覧照会
- `get_positions()` — 保有ポジション照会

services 層（`order_execution_pipeline.py` 等）はコンストラクターで `BrokerBase` 型の
引数を受け取り、具体的な実装クラスを知らない。

## 理由

- **具体クラス直接参照を採用しない理由**: services 層に `KabuBroker` などを直接 import すると、
  ペーパートレードへの切り替えや証券会社変更のたびにビジネスロジックを修正する必要が生じる。
  また、ユニットテストでモックが作れずテスト困難になる。
- **ファクトリー関数を採用しない理由**: 文字列ベースの分岐は型安全でなく、
  新しい Broker を追加した際にファクトリーの修正漏れが起きやすい。
  テスト用モックの注入も煩雑になる。
- **ABC + DI を採用する理由**:
  - services 層が `BrokerBase` 型のみに依存するため、Broker の変更が上位層に伝播しない（依存性逆転）。
  - テスト時に `BrokerBase` を実装したモック Broker を注入でき、外部 API なしでユニットテストが書ける。
  - 新しい証券会社への対応は `BrokerBase` を継承した新クラスを追加するだけで、既存コードを変更しない（OCP）。

## 結果

**正の効果**:
- `PaperBroker` と `KabuBroker` を設定やコンストラクター引数で切り替えられる。
- services 層のユニットテストで外部 API への依存を除去できる。
- 新しい証券会社対応を独立して追加できる。

**負の効果（トレードオフ）**:
- `BrokerBase` が抽象化しすぎると、証券会社固有の機能（例: Kabu 特有の注文オプション）を
  汎用インターフェースに押し込む設計上の摩擦が生じる可能性がある。
- DI コンテナを使わない手動 DI のため、Broker の生成・注入は呼び出し側（`run_*.py` や
  スケジューラー）が責任を持つ必要がある。

## 関連

- `python/src/trading/brokers/base.py` — `BrokerBase` 実装
- `python/src/trading/brokers/paper/` — PaperBroker 実装
- `python/src/trading/brokers/kabu/` — KabuBroker 実装
- `docs/ARCHITECTURE.md` — brokers 層の説明
