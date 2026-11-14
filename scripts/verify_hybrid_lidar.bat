@echo off
setlocal DisableDelayedExpansion
if not defined BEVFUSION_WSL_DISTRO set "BEVFUSION_WSL_DISTRO=Ubuntu-26.04"
pushd "%~dp0.."
if errorlevel 1 exit /b 1
rem Run both verification and optional plugin export in the WSL Conda environment.
wsl.exe --distribution %BEVFUSION_WSL_DISTRO% --exec bash scripts/verify_hybrid_lidar.sh %*
set "VERIFY_RESULT=%errorlevel%"
popd
exit /b %VERIFY_RESULT%
