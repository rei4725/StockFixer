@echo off
rem UI だけの依存（本番イメージと CI には入れない）。streamlit は pillow 13 未満を要るので別の venv に requirements-ui.txt と pillow を除いた requirements.txt を入れてから使う（手順は requirements-ui.txt の頭）
cd /d %~dp0
py -m streamlit run run_backtest_ui.py
pause
