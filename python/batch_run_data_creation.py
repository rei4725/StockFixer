import csv
import subprocess
CSV_PATH = "python/データ取得対象.csv"
SCRIPT_PATH = "python/run_data_creation.py"

with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        market = row["市場"]
        symbol = row["銘柄コード"]
        cmd = [
            "python",
            SCRIPT_PATH,
            "--market", market,
            "--symbol", symbol
        ]
        print("実行:", " ".join(cmd))
        subprocess.run(cmd, check=True)

# 全てのデータ取得が終わった後、モデル作成スクリプトを実行
print("全データ取得完了後、モデル作成スクリプトを実行します。")
subprocess.run(["python", "python/run_model_creation.py"], check=True)
