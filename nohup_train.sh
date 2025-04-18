#!/bin/bash

 # L-VITON with Large Image Encoder (= DINOv2)
nohup env CUDA_VISIBLE_DEVICES=3 python train.py \
 --config_name VITONHD \
 --transform_size shiftscale3 hflip \
 --transform_color hsv bright_contrast \
 --resume_path ./logs/base/models/[Train]_[epoch=19]_[train_loss_epoch=0.0390].ckpt \
 --resume_from_checkpoint ./logs/base/models/[Train]_[epoch=19]_[train_loss_epoch=0.0390].ckpt \
 --save_name L-VITON \
 --batch_size 16 \
 --data_root_dir ../viton/dataset/zalando-hd-resized \
 --seed 23 \
 --save_every_n_epochs 5 > inference.log 2>&1 &