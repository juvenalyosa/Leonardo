param(
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

Write-Host "Launching Leonardo through WSL2"
Write-Host "Repository: $RepoRoot"
Write-Host ""

$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) {
    Write-Host "ERROR: WSL is not installed. Run 'Install Leonardo Windows.bat' after installing WSL2." -ForegroundColor Red
    exit 1
}

$wslRepo = (& wsl.exe wslpath -a "$RepoRoot" 2>&1 | Select-Object -First 1).Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslRepo)) {
    Write-Host "ERROR: Could not convert repository path to a WSL path." -ForegroundColor Red
    exit 1
}

Write-Host "On Windows 11, PyMOL GUI should open through WSLg."
Write-Host "On Windows 10, an external X server is required."
Write-Host ""

$quotedRepo = $wslRepo.Replace("'", "'\''")
$cmd = "cd '$quotedRepo' && ./scripts/launch_leonardo.sh"
& wsl.exe -e bash -lc $cmd
exit $LASTEXITCODE
