#!/usr/bin/env bash


EXP_DIR=exps/OWOD_t1_ore_extending
PY_ARGS=${@:1}

python -u main_open_world.py \
  --output_dir ${EXP_DIR} --data_root './data/OWOD' \
  --dataset owod --num_queries 100 --eval_every 5 \
  --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 20 \
  --train_set 't1_train' --test_set 'all_task_test' \
  --num_classes 81 \
  --epochs 70 --featdim 1024 \
  --backbone 'dino_resnet50' \
  --pretrain 'exps/OWOD_t1_ore/checkpoint0049.pth' \
  --cooling_prev 50 \
  --cooling \
  ${PY_ARGS}





