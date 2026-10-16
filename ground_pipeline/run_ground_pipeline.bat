@echo off
chcp 65001 >nul 2>&1

wsl -e bash /mnt/g/UGit/Phy-SIC/ground_pipeline/run_ground_pipeline.sh
if errorlevel 1 (
    echo [FAIL] run failed
    pause
    exit /b 1
)
