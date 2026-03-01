"""データ構造確認スクリプト"""
import sys
sys.path.insert(0, '.')
import duckdb
import pandas as pd

con = duckdb.connect('data/stockfixer.duckdb', read_only=True)

# stock_features の構造確認
df = con.execute("SELECT * FROM stock_features WHERE market='us' AND symbol='KO' ORDER BY row_num LIMIT 5").fetchdf()
print("=== stock_features columns (first 20) ===")
print(list(df.columns)[:20])
close_cols = [c for c in df.columns if 'close' in c.lower()]
print(f"\nclose系列カラム: {close_cols[:8]}")
print(f"'y' 列あり: {'y' in df.columns}")
print(f"'Date' 列あり: {'Date' in df.columns}")
print(f"shape: {df.shape}")
print(f"\nrow_num range: {df['row_num'].min()} ～ {df['row_num'].max()}")
print(f"\n最初の行のrow_numとclose系カラム:")
show_cols = ['market', 'symbol', 'row_num'] + close_cols[:3] + (['y'] if 'y' in df.columns else [])
print(df[show_cols].head(3))

# market_data_raw も確認
raw_count = con.execute("SELECT COUNT(*) FROM market_data_raw").fetchone()[0]
print(f"\n=== market_data_raw 件数: {raw_count} ===")
if raw_count > 0:
    r = con.execute("SELECT market, symbol, COUNT(*) as n FROM market_data_raw GROUP BY market, symbol LIMIT 5").fetchall()
    for row in r:
        print(row)
