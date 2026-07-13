@echo off
setlocal
for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"

echo Launching Leonardo through WSL2
echo ==============================
echo.
echo PyMOL will open from the Leonardo conda environment inside WSL.
echo Inside PyMOL, use: Plugins ^> Leonardo
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\installers\windows\launch_leonardo_windows_wsl.ps1" -RepoRoot "%REPO_ROOT%"
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
  echo.
  echo Launch failed with exit code %EXITCODE%.
  echo If Leonardo was not installed yet, run "Install Leonardo Windows.bat" first.
  echo.
  pause
)
exit /b %EXITCODE%
