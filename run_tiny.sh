#!/bin/bash
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
module load gcc/5.4
source activate detrhn
export PYTHONUNBUFFERED=1

CUR=$(pwd)

GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 $CUR/configs/OWOD_tiny_split.sh

