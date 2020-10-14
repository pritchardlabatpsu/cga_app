#!/bin/bash

mkdir -p cluster_job

# Submit jobs to cluster
qsub -cwd -N sanger run05-model_rf_boruta-Sanger.sh
