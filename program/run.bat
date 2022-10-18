@echo off

:: UNCOMMENT (remove ::) if you want the backend to automatically save files for you
:: export SAVE_FILES="1"

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set VENV_DIR=venv)

set ERROR_REPORTING=FALSE

set PYTHON="%~dp0%VENV_DIR%\Scripts\Python.exe"

set DTYPE=float32
set CLIP_CONTEXTS=3
set AMP=1
set MODEL=stable-diffusion
set DEV=True
set MODEL_PATH=models/animefull-final-pruned
::these aren't actually used by the site?
::set MODULE_PATH=models/modules
::unclear if these are used either
::set PRIOR_PATH=models/vector_adjust_v2.pt
set ENABLE_EMA=1
set VAE_PATH=models/animevae.pt
set PENULTIMATE=1
set PYTHONDONTWRITEBYTECODE=1

%PYTHON% -m uvicorn --host 0.0.0.0 --port=6969 --workers 1 main:app
pause
