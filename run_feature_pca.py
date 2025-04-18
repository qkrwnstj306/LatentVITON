import argparse, os 
from tqdm import tqdm, tqdm
import torch
from utils import visualize_and_save_features_pca, visualize_and_save_features_tsne, visualize_and_save_features_umap
import numpy as np
from einops import rearrange

def load_experiments_features(feature_maps_path, t):
#     pth_files = [
#     f for f in os.listdir(feature_maps_path) 
#     if f.endswith(f"_{t}.pt")
#      ]   
    pth_files = [
    f for f in os.listdir(feature_maps_path) 
    if f.endswith(f".pt")
    ]   
    
    feature_maps = []
    file_names = []
    for i, pth_file in enumerate(pth_files):
        feature_map = torch.load(os.path.join(feature_maps_path, pth_file)) # [4, 64, 48]
        feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N (=h x w) X C > 3072 x 4
        feature_maps.append(feature_map)
        filename_no_ext = pth_file.removesuffix(".pt")
        file_names.append(filename_no_ext)

    return feature_maps, file_names

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--feature_pth",
        type=str,
        default="./feature_map"
    )

    parser.add_argument(
        "--reduce_method",
        type=str,
        default="pca"
    )

    parser.add_argument(
        "--time_step",
        type=int,
        default=0
    )

    opt = parser.parse_args()
    feature_pth = opt.feature_pth
    time_step = opt.time_step
    
    if opt.reduce_method == "pca":
        visualize_and_save_features = visualize_and_save_features_pca
    elif opt.reduce_method == "tsne":
        visualize_and_save_features = visualize_and_save_features_tsne
    elif opt.reduce_method == "umap":
        visualize_and_save_features = visualize_and_save_features_umap
    pca_folder_path = f"./PCA_features_vis/{opt.reduce_method}"    
    os.makedirs(pca_folder_path, exist_ok=True)

    fit_features, file_names = load_experiments_features(feature_pth, time_step)  # N X C
    transform_features, _ = load_experiments_features(feature_pth, time_step)
    visualize_and_save_features(torch.cat(fit_features, dim=0),
                                    torch.cat(transform_features, dim=0),
                                    time_step,
                                    pca_folder_path,
                                    file_names
                                    )
    
if __name__ == "__main__":
    main()