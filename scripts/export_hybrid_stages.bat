@echo off
setlocal DisableDelayedExpansion
if not defined BEVFUSION_WSL_DISTRO set "BEVFUSION_WSL_DISTRO=Ubuntu-26.04"
pushd "%~dp0.."
if errorlevel 1 exit /b 1
wsl.exe --distribution %BEVFUSION_WSL_DISTRO% --exec bash scripts/export_hybrid_stages.sh %*
set "EXPORT_RESULT=%errorlevel%"
popd
exit /b %EXPORT_RESULT%
