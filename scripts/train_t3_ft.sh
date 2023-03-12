#!/bin/sh
module load anaconda/2020.11
module load cuda/10.2
module load cudnn/8.1.0.77_CUDA10.2
module load gcc/5.4
source activate hainanmore
export PYTHONUNBUFFERED=1


python tools/train_net.py \
--num-gpus 8 \
--dist-url='auto' \
--config-file ./configs/OWOD/t3/t3_ft.yaml \
MODEL.WEIGHTS "./output/t3_ours/model_0199999.pth" \
SOLVER.IMS_PER_BATCH 8 \
SOLVER.BASE_LR 0.0001 \
OUTPUT_DIR "./output/t3_ours_ft" \
DATALOADER.NUM_WORKERS 4 \
SOLVER.MAX_ITER 50000 \
TEST.EVAL_PERIOD 10000 \
SOLVER.CHECKPOINT_PERIOD 10000 \
OWOD.ENABLE_CLUSTERING False \
OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False \

