#!/bin/bash

# nohup env CUDA_VISIBLE_DEVICES=0 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 16 \
#  --cfg_scale 3 \
#  --model_load_path ./ckpts/VITONHD.ckpt \
#  --save_dir ./999_unmasked_replacement_cfg_3 > inference_unmasked_replacement_cfg_3_1.log 2>&1 &

# --repaint

 
nohup env CUDA_VISIBLE_DEVICES=3 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 1 \
 --cfg_scale 3 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --mcg \
 --unpair \
 --save_dir ./999_unmasked_mcg_t_cfg_3 > inference_unmasked_mcg_t_cfg_3_2.log 2>&1 &