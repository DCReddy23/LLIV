@echo off
setlocal

REM Launch the Gradio UI using the GPU venv if available, else CPU venv.
cd /d "%~dp0"

set "PY="
if exist ".venv-gpu\Scripts\python.exe" set "PY=.venv-gpu\Scripts\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not defined PY (
  echo ERROR: Could not find .venv-gpu\Scripts\python.exe or .venv\Scripts\python.exe
  echo Create a venv first, then install deps.
  exit /b 1
)

set "CONFIG=configs\custom_eval.yml"
set "RESUME=ckpt\stage2\stage2_weight.pth.tar"
set "HOST=127.0.0.1"
set "PORT=7860"

REM Optional overrides:
REM   run_ui.bat [port]
REM   run_ui.bat [host] [port]
if not "%~2"=="" (
  set "HOST=%~1"
  set "PORT=%~2"
) else if not "%~1"=="" (
  set "PORT=%~1"
)

echo Using Python: %PY%
echo UI: http://%HOST%:%PORT%

"%PY%" app.py --config "%CONFIG%" --resume "%RESUME%" --host "%HOST%" --port %PORT%
