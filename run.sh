#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 \
    python main.py \
    --exp_name CAINv2_train \
    --dataset biology \
    --batch_size 2 \
    --test_batch_size 1 \
    --model cain \
    --depth 3 \
    --loss 1*L1 \
    --max_epoch 200 \
    --lr 0.0002 \
    --log_iter 100 \
    --mode train \
    --data_root data/CAIN_dataV2