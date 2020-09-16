#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cnp
python run04-Lxonly_reg_rf_boruta_all.py $1
