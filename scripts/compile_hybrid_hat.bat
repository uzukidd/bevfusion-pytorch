@echo off
setlocal DisableDelayedExpansion
if not defined BEVFUSION_WSL_DISTRO set "BEVFUSION_WSL_DISTRO=Ubuntu-26.04"
pushd "%~dp0.."
if errorlevel 1 exit /b 1
wsl.exe --distribution %BEVFUSION_WSL_DISTRO% --exec bash scripts/compile_hybrid_hat.sh %*
set "COMPILE_RESULT=%errorlevel%"
popd
exit /b %COMPILE_RESULT%
