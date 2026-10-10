@echo off
set PYTHONUTF8=1
set POWER_PROJECT_VARIANT=real
cd /d "%~dp0"
py -m pip install -r requirements.txt
if errorlevel 1 goto :error
py -m src.data --download --prepare
if errorlevel 1 goto :error
py -m src.train
if errorlevel 1 goto :error
py -m src.evaluate
if errorlevel 1 goto :error
py -m src.eda
if errorlevel 1 goto :error
echo.
echo HOAN THANH. Ket qua THAT nam trong reports\real va models\real.
pause
exit /b 0
:error
echo.
echo CO LOI. Neu loi tai du lieu, hay tai ZIP thu cong tu trang UCI.
echo Dat ZIP tai: data\raw\individual_household_electric_power_consumption.zip
echo Sau do bam run_real.bat lai.
pause
exit /b 1
