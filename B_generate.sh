#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python Biology_generate.py \
    --exp_name CAINv2_train \
    --dataset biology \
    --model cain \
    --mode test \
    --batch_size 1 \
    --test_batch_size 1 \
    --resume true \
    --data_root data/DL \
    --resume_exp CAINv2_train \