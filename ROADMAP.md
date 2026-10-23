# ROADMAP

会社（ai-company）は上から順に Issue に分けて進める。★の段は人間の作業で、会社は Issue にしない。
設計: 口座アプリの切り出し（ai-company の docs/superpowers/specs/2026-10-23-account-app-design.md）。
口座アプリ（rei4725/StockFixer-account、private）は会社から見えない。約束は contracts/account-api.json だけ。

## 口座アプリの切り出し

- [x] ★段0: sandbox-runner を作り、本体のコンテナから docker.sock を外す
- [x] ★段1: 口座アプリの骨組み（共有ネットワーク stockfixer-account、サービス名 account:8000）
- [ ] 段2: 本体に AccountGatewayBroker（src/trading/brokers/account/）を足す
  - BrokerBase を実装し、contracts/account-api.json の POST /v1/orders・GET /v1/positions・/v1/balance・/v1/orders を呼ぶ
  - 環境変数 ACCOUNT_GATEWAY_URL と ACCOUNT_GATEWAY_TOKEN が無ければ使わない（今の動きを変えない）
  - client_order_id は「日付-ジョブ-銘柄-売買」で決まる値にし、やり直しで二重に注文しない
  - 口座アプリが応えない・10秒で時間切れなら、その日の残りの注文を出さずに記録し、Discord で知らせる。PaperBroker や KabuBroker に戻らない
  - 約束のテスト: contracts/account-api.json から偽の口座アプリを作り、通した・断った（7つの理由の符号）・時間切れを確かめる
  - docker-compose.yml の stockfixer を外部ネットワーク stockfixer-account に入れる1行は★人間がマージする
  - 口座アプリは朝 9:05（JST）の精算の間、注文を待たせる。本体はこの時刻の前後に注文を出さない（10秒で時間切れなら出さずに記録する、の決まりで安全な側に倒れる）
- [ ] 段3: 影の運用（10営業日）
  - ACCOUNT_SHADOW=true のとき、run_daily_auto_order が今の PaperBroker に出すのと同じ注文を、口座アプリにも出す（口座アプリの答えで本体の動きを変えない）
  - 毎日 results/account_shadow/YYYY-MM-DD.json に、両方の注文・断り・約定・損益の差を書く
  - 合否は★人間が決める（食い違いが上限の違いか約定の値の違いで説明できること）
- [ ] 段4: 切り替え
  - 紙の売買を口座アプリに一本化し、KabuBroker・1日の損の上限・AUTO_TRADE_MODE の live の道を本体から消す
  - needs:account-app の Issue を見直し、本体の仕事に戻せるものを company:todo に戻す
  - 禁止リストの縮めは★人間が company.yml で行う
