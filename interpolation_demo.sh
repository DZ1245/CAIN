#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name CAINv2_train \
    --dataset biology \
    --model cain \
    --mode test \
    --start_epoch 999 \
    --batch_size 2 \
    --test_batch_size 1 \
    --resume true \
    --data_root data/CAIN_dataV2