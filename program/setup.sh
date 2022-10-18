#!/bin/bash
set -ex
virtualenv venv
. venv/bin/activate
pip install -r requirements.txt 