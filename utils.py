import json
import argparse

import numpy as np
import torch.nn.functional as F

import torch
from sklearn.decomposition import PCA
import os
from PIL import Image
from math import sqrt
from torchvision import transforms as T
from sklearn.manifold import TSNE
from umap import UMAP

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
def load_args(from_path, is_test=True):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(from_path, "r") as f:
        args.__dict__ = json.load(f)
    args.is_test = is_test
    if "E_name" not in args.__dict__.keys():
        args.E_name = "basic"
    return args   
def tensor2img(x):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    # x = (x+1)/2
    # x = np.clip(x, 0, 1)
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    x = np.uint8(x*255.0)
    if x.shape[-1] == 1:  # gray sclae
        x = np.concatenate([x,x,x], axis=-1)
    return x
def resize_mask(m, shape):
    m = F.interpolate(m, shape)
    m[m > 0.5] = 1
    m[m < 0.5] = 0
    return m

    
@torch.no_grad()
def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, t, save_dir, file_names):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(file_names), -1, 3)  # B x (H * W) x 3
    for i, file_name in enumerate(file_names):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        w = int(sqrt(pca_img.shape[0] * 3 / 4))
        h = int(w * 4 / 3)
        
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.BICUBIC)(pca_img)
        pca_img.save(os.path.join(save_dir, f"{file_name}.png"))
        
@torch.no_grad()
def visualize_and_save_features_tsne(feature_maps_fit_data, feature_maps_transform_data, t, save_dir, file_names):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    feature_maps_transform_data = feature_maps_transform_data.cpu().numpy()

    tsne = TSNE(n_components=3)
    feature_maps_tsne = tsne.fit_transform(feature_maps_transform_data)  # N x 3

    feature_maps_tsne = feature_maps_tsne.reshape(len(file_names), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(file_names):
        tsne_img = feature_maps_tsne[i]  # (H * W) x 3
        w = int(sqrt(tsne_img.shape[0] * 3 / 4))
        h = int(w * 4 / 3)
        
        tsne_img = tsne_img.reshape(h, w, 3)
        tsne_img_min = tsne_img.min(axis=(0, 1))
        tsne_img_max = tsne_img.max(axis=(0, 1))
        tsne_img = (tsne_img - tsne_img_min) / (tsne_img_max - tsne_img_min)
        tsne_img = Image.fromarray((tsne_img * 255).astype(np.uint8))
        tsne_img = T.Resize((512), interpolation=T.InterpolationMode.BICUBIC)(tsne_img)
        tsne_img.save(os.path.join(save_dir, f"batch-{experiment}_time_{t}.png"))
        
@torch.no_grad()
def visualize_and_save_features_umap(feature_maps_fit_data, feature_maps_transform_data, t, save_dir, file_names):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    feature_maps_transform_data = feature_maps_transform_data.cpu().numpy()

    umap = UMAP(n_components=3)
    feature_maps_umap = umap.fit_transform(feature_maps_transform_data)  # N x 3

    feature_maps_umap = feature_maps_umap.reshape(len(file_names), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(file_names):
        umap_img = feature_maps_umap[i]  # (H * W) x 3
        w = int(sqrt(umap_img.shape[0] * 3 / 4))
        h = int(w * 4 / 3)
        
        umap_img = umap_img.reshape(h, w, 3)
        umap_img_min = umap_img.min(axis=(0, 1))
        umap_img_max = umap_img.max(axis=(0, 1))
        umap_img = (umap_img - umap_img_min) / (umap_img_max - umap_img_min)
        umap_img = Image.fromarray((umap_img * 255).astype(np.uint8))
        umap_img = T.Resize((512), interpolation=T.InterpolationMode.BICUBIC)(umap_img)
        umap_img.save(os.path.join(save_dir, f"batch-{experiment}_time_{t}.png"))