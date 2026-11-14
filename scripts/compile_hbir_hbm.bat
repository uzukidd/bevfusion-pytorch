@echo off
setlocal DisableDelayedExpansion
rem The vendor compiler and plugin are Linux binaries. Path arguments after this launcher use WSL paths.
set "PLUGIN_DISTRO=%BEVFUSION_WSL_DISTRO%"
if not defined PLUGIN_DISTRO set "PLUGIN_DISTRO=Ubuntu-26.04"
set "PLUGIN_PYTHON=%BEVFUSION_WSL_PYTHON%"
if not defined PLUGIN_PYTHON set "PLUGIN_PYTHON=/home/administrator/miniconda3/envs/bevfusion-pytorch/bin/python"
set "PLUGIN_SCRIPT="
for /f "usebackq delims=" %%I in (`wsl.exe --distribution %PLUGIN_DISTRO% --exec wslpath -a -u "%~dp0compile_hbir_hbm.sh"`) do set "PLUGIN_SCRIPT=%%I"
if not defined PLUGIN_SCRIPT (
    >&2 echo Failed to resolve the export launcher in WSL.
    exit /b 1
)
wsl.exe --distribution %PLUGIN_DISTRO% --exec env "BEVFUSION_PYTHON=%PLUGIN_PYTHON%" bash "%PLUGIN_SCRIPT%" %*
exit /b %errorlevel%
