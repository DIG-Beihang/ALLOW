#!/bin/sh
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
module load gcc/5.4
source activate hainanmore
export PYTHONUNBUFFERED=1

DIR=$1

python tools/train_net.py \
--num-gpus 8 \
--dist-url='auto' \
--eval-only \
--resume \
--config-file ./configs/OWOD/t1/t1_test.yaml \
MODEL.WEIGHTS $DIR"/model_final.pth" \
SOLVER.IMS_PER_BATCH 8 \
SOLVER.BASE_LR 0.0001 \
OUTPUT_DIR $DIR \
DATALOADER.NUM_WORKERS 4 \
SOLVER.MAX_ITER 6000 \
TEST.EVAL_PERIOD 1000 \
