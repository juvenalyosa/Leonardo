@echo off
setlocal
cd /d "%~dp0"

echo Launching Leonardo through WSL2
echo ==============================
echo.
echo PyMOL will open from the Leonardo conda environment inside WSL.
echo Inside PyMOL, use: Plugins ^> Leonardo
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installers\windows\launch_leonardo_windows_wsl.ps1" -RepoRoot "%~dp0"
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
  echo.
  echo Launch failed with exit code %EXITCODE%.
  echo If Leonardo was not installed yet, run "Install Leonardo Windows.bat" first.
  echo.
  pause
)
exit /b %EXITCODE%
