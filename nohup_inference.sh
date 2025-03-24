#!/bin/bash

nohup env CUDA_VISIBLE_DEVICES=3 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --repaint \
 --save_dir ./s_l_VITON_repaint > inference.log 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=2 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 16 \
#  --model_load_path ./ckpts/VITONHD.ckpt \
#  --unpair \
#  --repaint \
#  --save_dir ./s_l_VITON_repaint > inference2.log 2>&1 &

# --repaint