#!/bin/bash

mkdir -p cluster_job

# Submit jobs to cluster
qsub -cwd -N sanger -o cluster_job/sanger_all -e cluster_job/sanger_all run05-model_rf_boruta-Sanger.sh
