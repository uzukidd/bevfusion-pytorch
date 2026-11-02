@echo off
setlocal DisableDelayedExpansion
call "%~dp0run_python.bat" python scripts/verify_bevfusion_minimal.py %*
exit /b %errorlevel%
