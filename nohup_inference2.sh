#!/bin/bash

if [ $1 == 1 ]; then
    nohup env CUDA_VISIBLE_DEVICES=2 python lm_inference.py \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --code_name "pure" \
    --denoise_steps 51 \
    --save_dir ./999_pure51step_cfg1 > inference_pure51step_cfg1_1.log 2>&1 &

elif [ $1 == 2 ]; then
    nohup env CUDA_VISIBLE_DEVICES=2 python lm_inference.py \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --code_name "pure" \
    --denoise_steps 51 \
    --unpair \
    --save_dir ./999_pure51step_cfg1 > inference_pure51step_cfg1_2.log 2>&1 &

elif [ $1 == 3 ]; then
    nohup env CUDA_VISIBLE_DEVICES=3 python lm_inference.py \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --code_name "unmasked" \
    --use_pure_to_prior \
    --apply_lm \
    --save_dir ./999_UnmaskedDdpmPrior_woNoise_hardFFT_woStochastic_cfg1 > inference_UnmaskedDdpmPrior_woNoise_hardFFT_woStochastic_cfg1_1.log 2>&1 &

elif [ $1 == 4 ]; then
    nohup env CUDA_VISIBLE_DEVICES=3 python lm_inference.py \
    --config_path ./configs/VITONHD.yaml \
    --batch_size 1 \
    --cfg_scale 1 \
    --model_load_path ./ckpts/VITONHD.ckpt \
    --code_name "unmasked" \
    --use_pure_to_prior \
    --apply_lm \
    --unpair \
    --save_dir ./999_UnmaskedDdpmPrior_woNoise_hardFFT_woStochastic_cfg1 > inference_UnmaskedDdpmPrior_woNoise_hardFFT_woStochastic_cfg1_2.log 2>&1 &
fi