#### paired
CUDA_VISIBLE_DEVICES=3 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --save_dir ./results

#### unpaired
CUDA_VISIBLE_DEVICES=4 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --unpair \
 --save_dir ./results

#### paired repaint
CUDA_VISIBLE_DEVICES=4 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --repaint \
 --save_dir ./results

#### unpaired repaint
CUDA_VISIBLE_DEVICES=4 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --unpair \
 --repaint \
 --save_dir ./results

 #### paired -> extract feature map per time step -> pca
CUDA_VISIBLE_DEVICES=0 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --img_callback

CUDA_VISIBLE_DEVICES=1 python run_feature_pca.py \
 --reduce_method pca \
 --time_step 0 \
 --feature_pth ./feature_map


 #### paired, predicted x_0
CUDA_VISIBLE_DEVICES=0 python inference.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \
 --predicted_x0

 #### Extract Masked Feature Statistics
CUDA_VISIBLE_DEVICES=3 python masked_feature_collect.py \
 --config_path ./configs/VITONHD.yaml \
 --batch_size 16 \
 --model_load_path ./ckpts/VITONHD.ckpt \