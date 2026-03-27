@echo off
cd /d %~dp0
py -m streamlit run run_backtest_ui.py
pause
