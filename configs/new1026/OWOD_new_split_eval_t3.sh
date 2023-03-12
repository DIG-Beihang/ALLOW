#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t3_ore_cooling
PY_ARGS=${@:1}

python -u main_open_world.py \
    --output_dir ${EXP_DIR} --data_root './data/OWOD' --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_ft' --test_set 'all_task_test' --num_classes 81 \
    --epochs 310 --featdim 1024 \
    --backbone 'dino_resnet50' \
    --resume 'exps/OWOD_t3_ore_cooling/checkpoint0309.pth' --eval \
    ${PY_ARGS}
