#!/bin/sh
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
module load gcc/5.4
source activate hainan
export PYTHONUNBUFFERED=1


python tools/train_net.py \
--num-gpus 4 \
--dist-url='auto' \
--resume \
--config-file ./configs/OWOD/t1/t1_train.yaml \
MODEL.WEIGHTS "./pretrained/R-50.pkl" \
SOLVER.IMS_PER_BATCH 8 \
SOLVER.BASE_LR 0.01 \
OUTPUT_DIR "./output/t1" \
DATALOADER.NUM_WORKERS 4 \
SOLVER.MAX_ITER 50000 \
TEST.EVAL_PERIOD 5000 \
SOLVER.CHECKPOINT_PERIOD 5000 \
OWOD.ENABLE_CLUSTERING False \
OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False \
