#!/bin/bash
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
module load gcc/5.4
source activate detrhn
export PYTHONUNBUFFERED=1

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 configs/new1026/OWOD_ore_split_t2_extending.sh
