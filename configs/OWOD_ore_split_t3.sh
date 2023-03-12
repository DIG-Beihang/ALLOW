#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t3_new_split
PY_ARGS=${@:1}

python -u main_open_world.py \
  --output_dir ${EXP_DIR} --data_root /data/home/scv6140/run/oln_owod-oln_owod/oln_owod-oln_owod/datasets \
  --dataset owod --num_queries 100 --eval_every 5 \
  --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 \
  --train_set 'owod_t3_train' --test_set 'owod_all_task_test' \
  --num_classes 81 \
  --unmatched_boxes --epochs 200 --lr 2e-5 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
  --backbone 'dino_resnet50' \
  --pretrain 'exps/OWOD_t3_ft/checkpoint.pth' \
  --eval-only \
  ${PY_ARGS}




