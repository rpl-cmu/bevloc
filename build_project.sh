#! /bin/bash

# Get gdal and python 3.8
apt-get install libgdal-dev python3.8-dev -y
pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
python3 -m poetry add "GDAL==$(gdal-config --version)"

# Setup the project with poetry
pip install poetry
python3 -m poetry config virtualenvs.in-project true # puts .venv in the current directory
python3 -m poetry lock --no-update 
python3 -m poetry install