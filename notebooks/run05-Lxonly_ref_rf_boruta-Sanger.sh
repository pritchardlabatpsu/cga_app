#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate cnp
python run05-Lxonly_ref_rf_boruta-Sanger.py $1
