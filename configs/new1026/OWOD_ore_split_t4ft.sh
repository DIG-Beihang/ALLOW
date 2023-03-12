#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t4_ore_ft
PY_ARGS=${@:1}

python -u main_open_world.py \
  --output_dir ${EXP_DIR} --data_root './data/OWOD' \
  --dataset owod --num_queries 100 --eval_every 5 \
  --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 \
  --train_set 't4_ft' --test_set 'all_task_test' \
  --num_classes 81 \
  --epochs 400 --featdim 1024 \
  --backbone 'dino_resnet50' \
  --pretrain 'exps/OWOD_t4_ore/checkpoint0359.pth' \
  ${PY_ARGS} 




