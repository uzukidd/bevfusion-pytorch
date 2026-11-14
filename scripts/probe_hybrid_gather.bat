@echo off
setlocal DisableDelayedExpansion
rem Horizon compiler binaries run inside WSL; path arguments use WSL paths.
set "PLUGIN_DISTRO=%BEVFUSION_WSL_DISTRO%"
if not defined PLUGIN_DISTRO set "PLUGIN_DISTRO=Ubuntu-26.04"
set "PLUGIN_PYTHON=%BEVFUSION_WSL_PYTHON%"
if not defined PLUGIN_PYTHON set "PLUGIN_PYTHON=/home/administrator/miniconda3/envs/bevfusion-pytorch/bin/python"
set "PLUGIN_SCRIPT="
for /f "usebackq delims=" %%I in (`wsl.exe --distribution %PLUGIN_DISTRO% --exec wslpath -a -u "%~dp0probe_hybrid_gather.sh"`) do set "PLUGIN_SCRIPT=%%I"
if not defined PLUGIN_SCRIPT (
    >&2 echo Failed to resolve the probe launcher in WSL.
    exit /b 1
)
wsl.exe --distribution %PLUGIN_DISTRO% --exec env "BEVFUSION_PYTHON=%PLUGIN_PYTHON%" bash "%PLUGIN_SCRIPT%" %*
exit /b %errorlevel%
