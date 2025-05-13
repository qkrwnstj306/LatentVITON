#!/bin/bash

nohup env CUDA_VISIBLE_DEVICES=0 python lm_inference.py \
 --data_root_dir ./dataset/zalando-hd-resized \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --cfg_scale 1 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --apply_lm \
 --code_name "unmasked" \
 --unpair \
 --save_dir ./results > inference.log 2>&1 &