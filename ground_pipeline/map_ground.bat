@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

:: 可自定义要映射的端口（空格分隔）
set "PORTS=8001"
echo ======================================
echo [WSL Port Mapping Tool - ground_pipeline] %date% %time%
echo ======================================

:: 步骤1：管理员权限校验
echo [1/4] Checking admin privilege...
net session >nul 2>&1
if !errorLevel! neq 0 (
    powershell -Command "Start-Process '%~f0' -Verb RunAs" >nul 2>&1
    exit /b
)
echo [1/4] Admin privilege: OK

:: 步骤2：自动提取WSL IP（优先使用 hostname -I）
echo [2/4] Extracting WSL IP address...
set "WSL_IP="

:: 方法1: 使用临时文件获取 hostname -I 输出
wsl hostname -I > "%TEMP%\wsl_ip.txt" 2>nul
if exist "%TEMP%\wsl_ip.txt" (
    set /p WSL_HOST_OUTPUT=<"%TEMP%\wsl_ip.txt"
    del "%TEMP%\wsl_ip.txt"
    :: 提取第一个IP（处理空格分隔的多个IP）
    for /f "tokens^=1 delims^= " %%a in ("!WSL_HOST_OUTPUT!") do (
        set "WSL_IP=%%a"
    )
)

:: 方法2: 如果方法1失败，使用临时文件获取 ip addr 输出
if not defined WSL_IP (
    wsl -e bash -c "ip addr show eth0 2>&1 | grep -E 'inet [0-9]' | head -1 | awk '{print $2}' | cut -d/ -f1" > "%TEMP%\wsl_ip2.txt" 2>nul
    if exist "%TEMP%\wsl_ip2.txt" (
        set /p WSL_IP=<"%TEMP%\wsl_ip2.txt"
        del "%TEMP%\wsl_ip2.txt"
    )
)

:: 兜底IP（请根据实际情况修改）
if not defined WSL_IP (
    set "WSL_IP=172.19.35.67"
    echo [WARNING] Auto-detection failed, using fallback
)

echo [2/4] WSL IP: !WSL_IP!

:: 步骤3：清理旧映射规则
echo [3/4] Cleaning old port mapping rules...
for %%p in (!PORTS!) do (
    netsh interface portproxy delete v4tov4 listenport=%%p listenaddress=0.0.0.0 >nul 2>&1
    echo     - Cleaned port %%p
)

:: 步骤4：创建新的端口映射规则
echo [4/4] Creating new port mapping rules...
for %%p in (!PORTS!) do (
    netsh interface portproxy add v4tov4 listenport=%%p listenaddress=0.0.0.0 connectaddress=!WSL_IP! connectport=%%p
    if !errorLevel! equ 0 (
        echo     [OK] Windows:0.0.0.0:%%p -^> WSL:!WSL_IP!:%%p
    ) else (
        echo     [FAIL] Port %%p
    )
)

:: 显示最终结果
echo.
echo ==========================================
echo Current Port Proxy Rules:
echo ==========================================
netsh interface portproxy show v4tov4
echo.
echo ==========================================
echo Summary:
echo - WSL IP: !WSL_IP!
echo - Ports: !PORTS!
echo - Test: ping !WSL_IP!
echo ==========================================
echo.
echo Quick commands:
echo - View all rules: netsh interface portproxy show v4tov4
echo - Delete all: for %p in (!PORTS!) do netsh interface portproxy delete v4tov4 listenport=%p listenaddress=0.0.0.0
echo ==========================================
pause
exit /b 0
