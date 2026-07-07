@echo off
setlocal
cd /d "%~dp0"

echo Leonardo Windows installer
echo ==========================
echo.
echo This installer runs Leonardo inside WSL2 Ubuntu. The full scientific stack
echo is installed in Linux because several Leonardo dependencies are not robust
echo native Windows packages.
echo.
echo If WSL2 is not installed, run this once from Administrator PowerShell:
echo   wsl --install -d Ubuntu
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installers\windows\install_leonardo_windows_wsl.ps1" -RepoRoot "%~dp0"
set EXITCODE=%ERRORLEVEL%

echo.
if "%EXITCODE%"=="0" (
  echo Installation finished successfully.
  echo Next: double click "Launch Leonardo Windows.bat".
) else (
  echo Installation failed with exit code %EXITCODE%.
  echo Check the newest log in:
  echo   %~dp0install_logs
)
echo.
pause
exit /b %EXITCODE%
