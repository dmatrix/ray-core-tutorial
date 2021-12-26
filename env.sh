#!/usr/bin/env bash

# usage:
#   % source env.sh ray-core
#   % source env.sh ray-core 3.8.10
#
# Creates a conda env called "ray-core"
# with specified Python version

ENV_NAME=$1
PYTHON_VERSION=$2

if [ -z "$1" ]
  then
    echo "No environment named supplied!"
fi

if [ -z "$2" ]
  then
    PYTHON_VERSION=3.7.7
fi

conda deactivate
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME"
pip install -r requirements.txt
