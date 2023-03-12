#!/bin/bash

set -x

EXP_DIR=exps/OWOD_t1_new_split
PY_ARGS=${@:1}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --train_set 't1_train_new_split' --test_set 'test' --num_classes 81 \
    --epochs 50 --featdim 1024 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --train_set 't1_train_new_split' --test_set 'test' --num_classes 81 \
    --epochs 80 --featdim 1024 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t1_new_split/checkpoint0049.pth' \
    --cooling \
    --cooling_prev 50 \
    ${PY_ARGS}

:<<!
EXP_DIR=exps/OWOD_t2_new_split
PY_ARGS=${@:1}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --train_set 't2_train_new_split' --test_set 'test' --num_classes 81 \
    --epochs 100 --lr 2e-5 --featdim 1024 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t1_new_split/checkpoint0049.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t2_ft_new_split
PY_ARGS=${@:1}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 19 --CUR_INTRODUCED_CLS 21 --train_set 't2_ft_new_split' --test_set 'test' --num_classes 81 \
    --epochs 150 --featdim 1024 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t2_new_split/checkpoint0099.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t3_new_split
PY_ARGS=${@:1}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_train_new_split' --test_set 'test' --num_classes 81 \
    --epochs 200 --lr 2e-5 --featdim 1024 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t2_ft_new_split/checkpoint0149.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t3_ft_new_split
PY_ARGS=${@:1}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_ft_new_split' --test_set 'test' --num_classes 81 \
    --epochs 250 --featdim 1024  \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t3_new_split/checkpoint0199.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t4_new_split
PY_ARGS=${@:1}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --train_set 't4_train_new_split' --test_set 'test' --num_classes 81 \
    --epochs 300 --lr 2e-5  --featdim 1024 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t3_ft_new_split/checkpoint0249.pth' \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t4_ft_new_split
PY_ARGS=${@:1}

python -u main_open_world.py --data_root './data/OWDETR' \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 5 \
    --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 20 --train_set 't4_ft_new_split' --test_set 'test' --num_classes 81 \
    --epochs 350 --featdim 1024 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t4_new_split/checkpoint0299.pth' \
    ${PY_ARGS}
!
