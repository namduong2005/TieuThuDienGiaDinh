@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 goto :error
py -m src.data --demo
if errorlevel 1 goto :error
py -m src.train
if errorlevel 1 goto :error
py -m src.evaluate
if errorlevel 1 goto :error
echo.
echo HOAN THANH. Xem ket qua trong reports\figures.
pause
exit /b 0
:error
echo.
echo CO LOI. Hay chup lai noi dung cua so nay.
pause
exit /b 1
