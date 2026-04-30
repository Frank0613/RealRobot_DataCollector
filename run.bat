@echo off
setlocal

set ISAAC_PYTHON="D:\isaacsim\python.bat"


set PROJECT_DIR=%~dp0
set MAIN_SCRIPT="%PROJECT_DIR%main.py"


echo [INFO] Launching Franka Controller...
echo [INFO] Using Isaac Sim Python: %ISAAC_PYTHON%

call %ISAAC_PYTHON% %MAIN_SCRIPT% %*

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Simulation exited with error code %ERRORLEVEL%
    pause
) else (
    echo [INFO] Simulation finished.
)