## ファイル命名規則のTips
- 命名は一貫性を重視し、スネークケースを推奨（例：user_profile.md）。
- 拡張子は用途に応じて適切に使い分ける（例：.md, .json, .yaml）。

## Python開発環境Tips (Windows)
- Pythonコマンドは Python の部分を py に置換して使用する。
- 仮想環境の作成: py -m venv .venv
- ライブラリのインストール: cmd /c ".\.venv\Scripts\python.exe -m pip install -r requirements.txt"
- PowerShellでのパス指定が不安定な場合、cmd /c を利用すると安定する。
- Windows環境では、パスの区切り文字に \ を使用する。

## Pythonモジュール管理とインポート
- モジュール化により、コードの再利用性と可読性が向上する。
- モジュール参照: from モジュール名.ファイル名 import 関数名
- インポートエラー解決策:
    - 相対パス、絶対パスでのインポートを試す。
    - __init__.py ファイルの存在を確認する。
    - 環境変数の設定を確認する。
    - 仮想環境が有効になっているか確認する。
    - pip install が完了しているか確認する。
    - py コマンドで実行する。
    - sys.modules を操作してモジュールをモックする（最終手段）。

## データ取得ライブラリ
- **yfinance**:
    - 株価だけでなく、為替レート（例: JPY=X でUSD/JPY）も取得可能。
    - 安定して多様な金融商品データを取得できるため、第一候補として検討する。
- **ccxt**:
    - 暗号資産取引所APIのラッパーライブラリ。
    - 法定通貨ペア(FX)の取得は、対応している取引所を選ぶ必要がある（例: binance は非対応, bitfinex も限定的）。
    - 為替レート取得が主目的の場合、yfinance の方がシンプルで確実。

## ユニットテストのTips
- unittest.mock を使用して外部依存性を持つクラスをテストする。
- MagicMock を使用する際に、元のメソッドのロジックを保持するために side_effect を設定する。
- Pythonの環境パスの問題で、pip install が完了してもモジュールが見つからない場合がある。

## SeleniumとWebDriverのTips
- Seleniumを使用する際は、対応するブラウザのWebDriver（例: Chrome Driver）が必要。
- WebDriverはシステムのPATHに設定するか、コード内で明示的にパスを指定する必要がある。
- ヘッドレスモードで実行することで、GUIなしでブラウザ操作が可能。
- WebDriverWait と expected_conditions を使用して、要素のロードを待機することで安定したスクレイピングが可能。
- SBI証券のようなサイトでは、ログイン後の要素IDやクラス名が変更される可能性があるため、定期的な確認が必要。

### ChromeDriverセットアップ手順（Windows）
1. Chromeのバージョンを確認し、[公式サイト](https://chromedriver.chromium.org/downloads)から対応するChromeDriverをダウンロード。
2. ダウンロードしたchromedriver.exeを任意のディレクトリ（例: C:\tools\chromedriver）に配置。
3. chromedriver.exeのパスをシステム環境変数PATHに追加する。
   - もしくは、Pythonコード内で `webdriver.Chrome(executable_path="C:/tools/chromedriver/chromedriver.exe")` のように明示的にパスを指定する。
4. コマンドプロンプトで `chromedriver --version` を実行し、認識されているか確認。
5. Selenium実行時にバージョン不一致エラーが出る場合は、Chrome本体とChromeDriverのバージョンを揃える。

---

## yfinanceのマルチインデックス仕様とフラット化Tips
- yfinanceで複数銘柄やgroup_by="ticker"オプションを指定すると、DataFrameのカラムがタプル型（MultiIndex）になる。
    - 例: ('Close', 'AAPL'), ('Open', 'AAPL')
- 単一銘柄でもgroup_by="ticker"がデフォルトで有効な場合、('Close', 'AAPL')のようなカラムになる。
- このままだと「df['Close']」のような通常アクセスができないため、カラム名を1段にフラット化する必要がある。
- フラット化例（Pythonコード）:
    ```python
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns.values]
    ```
- フラット化により、後続の特徴量生成や保存処理がシンプルかつ安全になる。

---

- サブディレクトリを極力排除し、各主要.pyファイルを1階層に集約すると可読性・保守性が向上する。
- importパスも1階層構成に統一することで、パス修正やリファクタ時の影響範囲が明確になる。
- テストやスクリプトも新パスに合わせて修正すること。

## 株価データ特徴量拡張のTips
- technical_analysis.py の create_basic_lag_features を拡張し、feature_cols=None で全数値列（OHLCV＋テクニカル指標）にラグ特徴量を自動付与できるようにした。
- data_saver.py で add_technical_indicators を呼び出し、RSI・MACD・EMA・ATR などの指標を追加した上で、全数値列のラグ特徴量を生成することで、モデル入力の情報量を増やせる。
- 特徴量名の正規化処理も忘れずに行う。

## 出力ファイルパス・ファイル名設計Tips
- データ出力時は `[market]_[symbol]` サブディレクトリを自動生成し、その中にcsvを保存することで管理性を向上させる。
- ファイル名は `features_YYYY_MM_DD_YYYY_MM_DD.csv` 形式とし、用途（特徴量データ）と期間を明示する。
- サブディレクトリ＋用途明示ファイル名により、複数市場・銘柄・期間のデータ混在時も誤保存・誤読込を防止できる。

## データディレクトリ再編のTips
- 生データ(csv等)は `python/data/`（src外、プロジェクト直下）に集約することで管理が明確になる。
- データロジック（data_loader.py, data_saver.py等）は `python/src/data/logic/` にまとめるとimportパスが統一しやすい。
- importパス修正例: `from src.data.logic.data_saver import ...`
- パッケージとして `__init__.py` を配置しておくとimportエラー防止になる。

## モデル作成・保存・ロード・予測の知見
- ModelManagerの `load_model` はロードしたインスタンスを返す必要がある（return model_instance）。
- joblibで保存・ロードしたXGBoostモデルは型情報も保持されるため、ロード後も predict が利用できる。
- モデルファイルは `python/models/モデル名.joblib` に保存される。
- py コマンドでの実行や PYTHONPATH の設定でパス問題を回避できる。
- モデル作成・保存スクリプトの期間指定ロジックは、end_dateを現在日時から自動取得し、start_dateを5年前に設定することで柔軟に運用できる。
- 予測値は「直近データに対する翌営業日の終値」である。

---

## 🕒 更新履歴

- 2025-08-11 17:43：yfinanceのマルチインデックス仕様とフラット化Tipsを追記
- 2025-08-10 作成：初期ルールセット定義
- 2025-08-11 更新：内容を整理し、重複をまとめた
- 2025-08-11 追加：モデル作成・保存・ロード・予測に関する知見を追記
- 2025-08-11 16:07：期間指定ロジック・予測値の意味を追記
- 2025-08-11 16:58：株価データ特徴量拡張のTipsを追加
