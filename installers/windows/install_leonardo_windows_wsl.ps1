param(
    [string]$RepoRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

$LogDir = Join-Path $RepoRoot "install_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("install_windows_wsl_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $LogFile -Append | Out-Null

try {
    Write-Host "Leonardo Windows installer through WSL2"
    Write-Host "Repository: $RepoRoot"
    Write-Host "Log file:   $LogFile"
    Write-Host ""

    $wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
    if (-not $wsl) {
        Write-Host "ERROR: WSL is not installed or wsl.exe is not on PATH." -ForegroundColor Red
        Write-Host ""
        Write-Host "Install WSL2 first from an Administrator PowerShell:"
        Write-Host "  wsl --install -d Ubuntu"
        Write-Host ""
        Write-Host "Restart Windows if requested, open Ubuntu once to create the Linux user,"
        Write-Host "then run this installer again."
        exit 1
    }

    $statusText = (& wsl.exe --status 2>&1) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: WSL is installed but no usable default Linux distribution is configured." -ForegroundColor Red
        Write-Host ""
        Write-Host "Recommended setup:"
        Write-Host "  wsl --install -d Ubuntu"
        Write-Host ""
        Write-Host "Then open Ubuntu once, create your Linux user, and rerun this installer."
        Write-Host ""
        Write-Host $statusText
        exit 1
    }

    Write-Host "WSL status:"
    Write-Host $statusText
    Write-Host ""

    $wslRepo = (& wsl.exe wslpath -a "$RepoRoot" 2>&1 | Select-Object -First 1).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($wslRepo)) {
        Write-Host "ERROR: Could not convert repository path to a WSL path." -ForegroundColor Red
        Write-Host "Repository path: $RepoRoot"
        exit 1
    }

    Write-Host "WSL repository path: $wslRepo"
    Write-Host ""
    Write-Host "The Linux installer will run inside WSL. On Windows 11, PyMOL GUI should open through WSLg."
    Write-Host "On Windows 10, install and start an X server before launching PyMOL."
    Write-Host ""

    $quotedRepo = $wslRepo.Replace("'", "'\''")
    $cmd = "cd '$quotedRepo' && chmod +x scripts/install_leonardo.sh scripts/install_envs.sh scripts/install_pymol_loader.sh scripts/launch_leonardo.sh installers/linux/install_leonardo_linux.sh && ./installers/linux/install_leonardo_linux.sh"
    & wsl.exe -e bash -lc $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "WSL Leonardo installation failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Leonardo is installed in WSL."
    Write-Host "Launch from Windows by double clicking:"
    Write-Host "  $(Join-Path $RepoRoot 'installers\shortcuts\Launch Leonardo Windows.bat')"
    Write-Host ""
    Write-Host "Inside PyMOL open:"
    Write-Host "  Plugins > Leonardo"
}
finally {
    Stop-Transcript | Out-Null
}
