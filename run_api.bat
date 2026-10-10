@echo off
set PYTHONUTF8=1
set POWER_PROJECT_VARIANT=real
cd /d "%~dp0"
py -c "import fastapi, uvicorn, joblib, pandas, sklearn" >nul 2>&1
if errorlevel 1 (
  echo Dang cai cac thu vien can thiet lan dau...
  py -m pip install -r requirements.txt
  if errorlevel 1 goto :error
)
echo.
echo WEB dang chay tai: http://127.0.0.1:8000
echo Trang thu API:    http://127.0.0.1:8000/docs
echo Nhan Ctrl+C de dung API.
echo.
py -m uvicorn src.api:app --host 127.0.0.1 --port 8000
if errorlevel 1 goto :error
exit /b 0
:error
echo.
echo KHONG KHOI DONG DUOC API. Hay chup lai noi dung cua so nay.
pause
exit /b 1
