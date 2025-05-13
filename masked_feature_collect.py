import torch
import os
from os.path import join as opj
from omegaconf import OmegaConf
import argparse
from importlib import import_module
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # tqdm을 import

from ldm.models.diffusion.ddim import DDIMSampler
from cldm.model import create_model

from pytorch_lightning import seed_everything

def initialize_masked_region(latent, mask, stats_path="clothing_statistics.pth"):
    """ 
    저장된 통계를 이용하여 masked region(옷) 초기화
    - latent: (B, C, H, W)
    - mask: (B, 1, H, W)
    """
    stats = torch.load(stats_path, map_location="cpu")  # CPU에서 로드
    mean, std = stats["mean"], stats["std"]

    # Masked region을 저장된 통계를 기반으로 초기화
    noise = torch.randn_like(latent) * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    # torch.where을 사용하여 마스킹된 부분만 교체
    latent = torch.where(mask.bool().expand_as(latent), noise, latent)

    print("Masked region initialized.")
    return latent

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./dataset/zalando-hd-resized")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./masked_feature")

    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    return args

class InpaintingStatsCollector:
    def __init__(self):
        self.mean_list = []
        self.std_list = []

    def update(self, latent, mask):
        """ 
        - latent: (B, C, H, W)
        - mask: (B, 1, H, W) (1=masked, 0=unmasked)
        """

        masked_features = latent * mask
        
        mean = masked_features.sum(dim=[0, 2, 3]) / mask.sum(dim=[0, 2, 3])
        std = ((masked_features - mean[None, :, None, None])**2).sum(dim=[0, 2, 3]) / mask.sum(dim=[0, 2, 3])

        self.mean_list.append(mean)
        self.std_list.append(std)

    def get_statistics(self):
        """ 수집된 통계의 평균을 반환 """
        mean = torch.stack(self.mean_list).mean(dim=0)
        std = torch.stack(self.std_list).mean(dim=0)
        return mean.cpu(), std.cpu()  # GPU에 있을 경우 CPU로 변환하여 저장


@torch.no_grad()
def main(args):
    seed_everything(args.seed)
    
    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    config = OmegaConf.load(args.config_path)
    config.model.params.img_H = args.img_H
    config.model.params.img_W = args.img_W
    params = config.model.params

    model = create_model(config_path=None, config=config)
    load_cp = torch.load(args.model_load_path, map_location="cpu")
    load_cp = load_cp["state_dict"] if "state_dict" in load_cp.keys() else load_cp
    model.load_state_dict(load_cp)
    model = model.cuda()
    model.eval()

    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        is_paired=not args.unpair,
        is_test=False,
        is_sorted=True
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8) 
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    stats_collector = InpaintingStatsCollector()
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        print(f"{batch_idx}/{len(dataloader)}")
        # z: [BS, 4, 64, 48], c.keys(): 'c_crossattn', 'c_concat', 'first_stage_cond'
        # c_crossattn[0]: [BS, 3, 224, 244] for CLIP Image Encoder
        # c_concat[0]: [BS, 3, 512, 384] for ControlNet Input
        # first_stage_cond[BS]: [9, 64, 48] -> agnostic map: [4, 64, 48], resized mask: [1, 64, 48], pose: [4, 64, 48]
        z, c = model.get_input(batch, params.first_stage_key)
        
        agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        mask = 1. - agnostic_mask_inversion # Inpainting mask (옷 영역)

        stats_collector.update(z, mask)

    # 최종 통계 저장
    stats_path = opj(save_dir, "clothing_statistics.pth")
    mean_aggregated, std_aggregated = stats_collector.get_statistics()
    torch.save({"mean": mean_aggregated, "std": std_aggregated}, stats_path)
    print(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    args = build_args()
    main(args)
