#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t1_new_split_9w
PY_ARGS=${@:1}

python -u main_open_world.py \
  --output_dir ${EXP_DIR} --data_root /data/home/scv6140/run/oln_owod-oln_owod/oln_owod-oln_owod/datasets \
  --dataset owod --num_queries 100 --eval_every 5 \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
  --train_set '1026_t1_train' --test_set '1026_all_task_test' \
  --num_classes 81 \
  --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
  --backbone 'dino_resnet50' \
  ${PY_ARGS}




