#!/bin/bash

cd PyTorch-StudioGAN
CUDA_VISIBLE_DEVICES=0 python3 src/main.py \
    -e -l -stat_otf -c "./src/configs/ILSVRC2012/SAGAN.json" \
    --checkpoint_folder "../studiogan-configs/" --eval_type "test"
