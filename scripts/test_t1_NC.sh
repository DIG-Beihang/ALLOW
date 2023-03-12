#!/bin/bash
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
module load gcc/5.4
source activate detrhn
export PYTHONUNBUFFERED=1

GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 configs/new1026/OWOD_new_split_eval_t1_NC.sh
