#!/bin/bash

mkdir -p cluster_job

# Submit jobs to cluster
qsub -cwd -N sanger -o cluster_job/sanger_all.o -e cluster_job/sanger_all.e run05-model_rf_boruta-Sanger.sh
qsub -cwd -N l200_sanger -o cluster_job/L200_sanger_all.o -e cluster_job/L200_sanger_all.e run05-Lxonly_ref_rf_boruta-Sanger.sh 200
