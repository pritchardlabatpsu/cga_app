#!/bin/bash

mkdir -p cluster_job

# Submit jobs to cluster
qsub -cwd -N L100 -o cluster_job/o100 -e cluster_job/e100 run04-Lxonly_reg_rf_boruta_all.sh 100
qsub -cwd -N L200 -o cluster_job/o200 -e cluster_job/e200 run04-Lxonly_reg_rf_boruta_all.sh 200
