@echo off
powershell.exe -NonInteractive -ExecutionPolicy Bypass -File "%~dp0weekly_redeploy.ps1"
