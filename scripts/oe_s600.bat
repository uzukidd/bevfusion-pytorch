@echo off
setlocal DisableDelayedExpansion
if not defined OE_WSL_DISTRO set "OE_WSL_DISTRO=Ubuntu-26.04"
where wsl.exe >nul 2>&1
if errorlevel 1 (
    >&2 echo WSL is unavailable. Configure an x86_64 WSL2 Linux environment first.
    exit /b 1
)
pushd "%~dp0.."
if errorlevel 1 exit /b 1
wsl.exe --distribution %OE_WSL_DISTRO% --user root --exec bash scripts/oe_s600.sh %*
set "OE_RESULT=%errorlevel%"
popd
exit /b %OE_RESULT%
