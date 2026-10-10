@echo off
set PYTHONUTF8=1
set POWER_PROJECT_VARIANT=demo
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 goto :error
py -m src.data --demo
if errorlevel 1 goto :error
py -m src.train
if errorlevel 1 goto :error
py -m src.evaluate
if errorlevel 1 goto :error
py -m src.eda
if errorlevel 1 goto :error
echo.
echo HOAN THANH. Ket qua DEMO nam trong reports\demo va models\demo.
pause
exit /b 0
:error
echo.
echo CO LOI. Hay chup lai noi dung cua so nay.
pause
exit /b 1
