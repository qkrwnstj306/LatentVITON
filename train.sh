# VITONHD base
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
 --config_name VITONHD \
 --transform_size shiftscale3 hflip \
 --transform_color hsv bright_contrast \
 --save_name test \
 --data_root_dir ./dataset/DressCode_1024/upper \
 --batch_size 16


# # VITONHD ATVloss
# CUDA_VISIBLE_DEVICES=5,6 python train.py \
#  --config_name VITONHD \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --use_atv_loss \
#  --resume_path <first stage model path> \
#  --save_name ATVloss_test

# # L-VITON with Large Image Encoder (= DINOv2)
# CUDA_VISIBLE_DEVICES=3 python train.py \
#  --config_name VITONHD \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --resume_path ./ckpts/VITONHD.ckpt \
#  --save_name L-VITON \
#  --batch_size 16 \
#  --data_root_dir ../viton/dataset/zalando-hd-resized \
#  --seed 23
 