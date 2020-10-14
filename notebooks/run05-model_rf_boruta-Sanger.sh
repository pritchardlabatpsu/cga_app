#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cnp
python run05-model_rf_boruta-Sanger.py
