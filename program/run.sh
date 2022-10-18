#!/bin/bash

# UNCOMMENT if you want the backend to automatically save files for you
# export SAVE_FILES="1"

export DTYPE="float32"
export CLIP_CONTEXTS=3
export AMP="1"
export MODEL="stable-diffusion"
export DEV="True"
export MODEL_PATH="models/animefull-final-pruned"
#these aren't actually used by the site
#export MODULE_PATH="models/modules"
#unclear if these are used either
#export PRIOR_PATH="models/vector_adjust_v2.pt"
export ENABLE_EMA="1"
export VAE_PATH="models/animevae.pt"
export PENULTIMATE="1"
export PYTHONDONTWRITEBYTECODE=1

if [[ -f venv/bin/python ]]; then
    PYTHON=venv/bin/python
else
    PYTHON=python
fi

$PYTHON -m uvicorn --host 0.0.0.0 --port=6969 main:app
