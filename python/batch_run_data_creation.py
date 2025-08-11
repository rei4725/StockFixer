import csv
import subprocess

START_DATE = "2015-08-11"
END_DATE = "2025-08-11"
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
            "--symbol", symbol,
            "--start_date", START_DATE,
            "--end_date", END_DATE
        ]
        print("実行:", " ".join(cmd))
        subprocess.run(cmd, check=True)
