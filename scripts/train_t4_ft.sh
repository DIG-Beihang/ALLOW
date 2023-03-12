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
--config-file ./configs/OWOD/t4/t4_ft.yaml \
MODEL.WEIGHTS "./output/t4_ours_from_ft/model_0079999.pth" \
SOLVER.IMS_PER_BATCH 16 \
SOLVER.BASE_LR 0.001 \
OUTPUT_DIR "./output/t4_ours_ftfrom799993" \
DATALOADER.NUM_WORKERS 4 \
SOLVER.MAX_ITER 30000 \
TEST.EVAL_PERIOD 5000 \
SOLVER.CHECKPOINT_PERIOD 5000 \
OWOD.ENABLE_CLUSTERING False \
OWOD.ENABLE_THRESHOLD_AUTOLABEL_UNK False \

