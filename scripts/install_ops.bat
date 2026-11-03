@echo off
setlocal DisableDelayedExpansion
rem Install each selected CUDA operator as a pip package / 安装所选算子为 pip 包。
set "OPS_PYTHON=%BEVFUSION_PYTHON%"
if not defined OPS_PYTHON if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "OPS_PYTHON=%CONDA_PREFIX%\python.exe"
if not defined OPS_PYTHON if exist "D:\miniconda3\envs\vla-supervised\python.exe" set "OPS_PYTHON=D:\miniconda3\envs\vla-supervised\python.exe"
if not defined OPS_PYTHON (
    >&2 echo Activate a non-base Conda environment or set BEVFUSION_PYTHON.
    exit /b 1
)
"%OPS_PYTHON%" -c "import sys; from pathlib import Path; p=Path(sys.prefix); ok=(p/'conda-meta').is_dir() and p.name.lower() not in ('base','root','miniconda','miniconda3','anaconda','anaconda3','miniforge3'); sys.exit(0 if ok else 'Select a non-base Conda environment.')"
if errorlevel 1 exit /b 1
rem Discover the Conda CUDA toolkit without running obsolete VS2019 activation hooks.
for %%I in ("%OPS_PYTHON%") do set "OPS_PREFIX=%%~dpI"
if not defined CUDA_HOME if exist "%OPS_PREFIX%Library\bin\nvcc.exe" set "CUDA_HOME=%OPS_PREFIX%Library"
if not defined CUDA_HOME if defined CUDA_PATH set "CUDA_HOME=%CUDA_PATH%"
if not defined CUDA_HOME (
    >&2 echo CUDA Toolkit not found. Set CUDA_HOME to the directory containing bin\nvcc.exe.
    exit /b 1
)
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%OPS_PREFIX%Library\bin;%PATH%"
rem Initialize the installed x64 MSVC toolset when cl.exe is not on PATH.
where cl.exe >nul 2>&1
if errorlevel 1 (
    call :find_msvc
    if errorlevel 1 exit /b 1
)
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
rem Keep compiler diagnostics ASCII so PyTorch can decode them reliably.
set "VSLANG=1033"
if not defined MAX_JOBS set "MAX_JOBS=2"
set "OPERATOR=%~1"
if not defined OPERATOR set "OPERATOR=all"
if /I "%OPERATOR%"=="all" goto install_all
if /I "%OPERATOR%"=="voxel" goto install_one
if /I "%OPERATOR%"=="spconv" goto install_one
if /I "%OPERATOR%"=="bev_pool" goto install_one
>&2 echo Usage: install_ops.bat [all^|voxel^|spconv^|bev_pool]
exit /b 1
:install_all
for %%O in (bev_pool voxel spconv) do (
    call :install %%O
    if errorlevel 1 exit /b 1
)
exit /b 0
:install_one
call :install %OPERATOR%
exit /b %errorlevel%
:install
rem Reuse installed torch/dependencies; never replace them during extension builds.
echo Installing %1 with %OPS_PYTHON%
"%OPS_PYTHON%" -m pip install --no-build-isolation --no-deps "%~dp0..\ops\%1"
exit /b %errorlevel%
:find_msvc
set "OPS_VSROOT="
set "OPS_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%OPS_VSWHERE%" (
    >&2 echo Visual Studio C++ Build Tools not found. Use an x64 developer prompt.
    exit /b 1
)
for /f "usebackq delims=" %%V in (`"%OPS_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "OPS_VSROOT=%%V"
if not defined OPS_VSROOT (
    >&2 echo No Visual Studio installation with x64 C++ tools was found.
    exit /b 1
)
call "%OPS_VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
where cl.exe >nul 2>&1
exit /b %errorlevel%
