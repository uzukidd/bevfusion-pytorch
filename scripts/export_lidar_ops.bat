@echo off
setlocal DisableDelayedExpansion
set "EXPORT_PYTHON=%BEVFUSION_PYTHON%"
if not defined EXPORT_PYTHON if exist "D:\miniconda3\envs\vla-supervised\python.exe" set "EXPORT_PYTHON=D:\miniconda3\envs\vla-supervised\python.exe"
if not defined EXPORT_PYTHON (
    >&2 echo Set BEVFUSION_PYTHON to an existing non-base Conda Python with torch, onnx and onnxruntime.
    exit /b 1
)
"%EXPORT_PYTHON%" "%~dp0..\tools\export_lidar_ops.py" %*
exit /b %errorlevel%
