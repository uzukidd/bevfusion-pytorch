@echo off
setlocal DisableDelayedExpansion
set "RUN_PYTHON=%BEVFUSION_PYTHON%"
if defined RUN_PYTHON goto validate
if not defined CONDA_PREFIX goto fallback
if not defined CONDA_DEFAULT_ENV goto fallback
if exist "%CONDA_PREFIX%\pkgs\" goto fallback
if /I "%CONDA_DEFAULT_ENV%"=="base" goto fallback
if /I "%CONDA_DEFAULT_ENV%"=="root" goto fallback
if /I "%CONDA_DEFAULT_ENV%"=="default" goto fallback
if not exist "%CONDA_PREFIX%\conda-meta\" goto fallback
if exist "%CONDA_PREFIX%\python.exe" set "RUN_PYTHON=%CONDA_PREFIX%\python.exe"
if defined RUN_PYTHON goto validate
:fallback
set "RUN_PYTHON=D:\miniconda3\envs\vla-supervised\python.exe"
:validate
if not exist "%RUN_PYTHON%" (
    >&2 echo Python not found. Set BEVFUSION_PYTHON to an existing named Conda environment's python.exe.
    exit /b 1
)
"%RUN_PYTHON%" -c "import pathlib,sys; p=pathlib.Path(sys.prefix); sys.exit(0 if (p/'conda-meta').is_dir() and not (p/'pkgs').is_dir() and p.name.lower() not in ('base','root','default') else 1)"
if errorlevel 1 (
    >&2 echo Select an existing non-base Conda Python using BEVFUSION_PYTHON.
    exit /b 1
)
pushd "%~dp0.."
if errorlevel 1 exit /b 1
"%RUN_PYTHON%" test_seg.py %*
set "RUN_RESULT=%errorlevel%"
popd
exit /b %RUN_RESULT%
