import os
from os.path import join as opj
from omegaconf import OmegaConf
from importlib import import_module
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from cldm.model import create_model
from utils import tensor2img

from pytorch_lightning import seed_everything

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--model_load_path", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root_dir", type=str, default="./dataset/zalando-hd-resized")
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--unpair", action="store_true")
    parser.add_argument("--save_dir", type=str, default="./samples")

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--img_callback", action="store_true")
    args = parser.parse_args()
    return args


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

    '''Choose sampler among PLMS and DDIM'''
    #sampler = PLMSSampler(model)
    sampler = DDIMSampler(model)
    dataset = getattr(import_module("dataset"), config.dataset_name)(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        is_paired=not args.unpair,
        is_test=True,
        is_sorted=True
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H//8, img_W//8) 
    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        # z: [BS, 4, 64, 48], c.keys(): 'c_crossattn', 'c_concat', 'first_stage_cond'
        # c_crossattn[0]: [BS, 3, 224, 244] for CLIP Image Encoder
        # c_concat[0]: [BS, 3, 512, 384] for ControlNet Input
        # first_stage_cond[BS]: [9, 64, 48] -> agnostic map: [4, 64, 48], resized mask: [1, 64, 48], pose: [4, 64, 48]
        z, c = model.get_input(batch, params.first_stage_key)
        
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        mask = None
        # '''Alleviate SNR for x_T'''
        # start_code = model.q_sample(z, ts) 
        
        '''Orignal x_T'''
        C, H, W = shape
        size = (bs, C, H, W)
        start_code = torch.randn(size, device=z.device)
        
        # '''Alleviate SNR for x_T using only agnostic_mask'''
        # C, H, W = shape
        # size = (bs, C, H, W)
        # start_code = torch.randn(size, device=z.device)
        # alleviated_start_code = model.q_sample(z, ts) 
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        # start_code = start_code * (1. - agnostic_mask_inversion) + alleviated_start_code * agnostic_mask_inversion
        # mask = agnostic_mask_inversion

        # '''StableVITON + Ours'''
        # start_code = model.q_sample(z, ts) 
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        # mask = agnostic_mask_inversion
        
        img_callback_fn = None
        if args.img_callback:
            def save_feature_map(feature_map, filename):
                save_feature_dir = './feature_map'
                os.makedirs(save_feature_dir, exist_ok=True)
                save_path = os.path.join(save_feature_dir, f"{filename}.pt")
                torch.save(feature_map, save_path)
            def img_callback(img, time_step):
                for sample_idx, (sample, fn,  cloth_fn) in enumerate(zip(img, batch['img_fn'], batch["cloth_fn"])):
                    save_feature_map(sample, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}_{time_step}")
            img_callback_fn = img_callback


        samples, _, _ = sampler.sample(
            args.denoise_steps,
            bs,
            shape, 
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_conditioning=uc_full,
            mask=mask,
            x0=z,
            img_callback=img_callback_fn,
        )

        if args.img_callback:
            continue

        x_samples = model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if args.repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy()+1)/2 * 255)   # [0,255]
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1-repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)

            to_path = opj(save_dir, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:,:,::-1])

if __name__ == "__main__":
    args = build_args()
    main(args)
