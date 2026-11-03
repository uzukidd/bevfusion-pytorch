@echo off
setlocal DisableDelayedExpansion
set "ONNX_PYTHON=%BEVFUSION_PYTHON%"
if not defined ONNX_PYTHON if exist "D:\miniconda3\envs\vla-supervised\python.exe" set "ONNX_PYTHON=D:\miniconda3\envs\vla-supervised\python.exe"
if not defined ONNX_PYTHON (
    >&2 echo Set BEVFUSION_PYTHON to a Windows Conda environment with torch and onnxruntime.
    exit /b 1
)
"%ONNX_PYTHON%" "%~dp0..\test_onnx.py" %*
exit /b %errorlevel%
