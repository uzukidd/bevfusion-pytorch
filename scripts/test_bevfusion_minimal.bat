@echo off
setlocal DisableDelayedExpansion
if defined BEVFUSION_PYTHON set "WSLENV=BEVFUSION_PYTHON/u:%WSLENV%"
if defined BEVFUSION_CONDA_ENV set "WSLENV=BEVFUSION_CONDA_ENV/u:%WSLENV%"
set "RUN_SH="
for /f "usebackq delims=" %%I in (`wsl.exe --exec wslpath -a -u "%~dp0test_bevfusion_minimal.sh"`) do set "RUN_SH=%%I"
if not defined RUN_SH (
    >&2 echo Unable to resolve the launcher path in WSL.
    exit /b 1
)
wsl.exe --exec bash "%RUN_SH%" %*
exit /b %errorlevel%
