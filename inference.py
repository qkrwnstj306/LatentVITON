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
    parser.add_argument("--cfg_scale", type=float, default=1.0)

    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--img_H", type=int, default=512)
    parser.add_argument("--img_W", type=int, default=384)
    parser.add_argument("--eta", type=float, default=0.0)
    
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--img_callback", action="store_true")
    parser.add_argument("--predicted_x0", action="store_true")
    parser.add_argument("--sampling_schedule", type=str, default="uniform")
    parser.add_argument("--mcg", action="store_true")
    parser.add_argument("--dps", action="store_true")
    parser.add_argument("--pixel_mcg", action="store_true")
    parser.add_argument("--noisy_mcg", action="store_true")
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
        '''Alleviate SNR for x_T (StableVITON)'''
        # start_code = model.q_sample(z, ts) 
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        # mask = agnostic_mask_inversion
        
        '''Orignal x_T (Pure Gaussian noise)'''
        # C, H, W = shape
        # size = (bs, C, H, W)
        # start_code = torch.randn(size, device=z.device)
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        #mask = agnostic_mask_inversion
        
        '''Unmasked'''
        C, H, W = shape
        size = (bs, C, H, W)
        start_code = torch.randn(size, device=z.device)
        alleviated_start_code = model.q_sample(z, ts) 
        agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        start_code = start_code * (1. - agnostic_mask_inversion) + alleviated_start_code * agnostic_mask_inversion
        mask = agnostic_mask_inversion

        '''offset noise'''
        # C, H, W = shape
        # size = (bs, C, H, W)
        # start_code = torch.randn(size, device=z.device) + 0.1 * torch.randn((bs, C, 1, 1), device=z.device)
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        # mask = agnostic_mask_inversion
        
        
        '''Unmasked + offset noise'''
        # C, H, W = shape
        # size = (bs, C, H, W)
        # start_code = torch.randn(size, device=z.device) + 0.1 * torch.randn((bs, C, 1, 1), device=z.device)
        # alleviated_start_code = model.q_sample(z, ts) 
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        # start_code = start_code * (1. - agnostic_mask_inversion) + alleviated_start_code * agnostic_mask_inversion
        #mask = agnostic_mask_inversion

        """Unmasked + Masked Feature Statistics"""
        # def initialize_masked_region(latent, mask, stats_path="./masked_feature/clothing_statistics.pth"):
        #     """ 
        #     저장된 통계를 이용하여 masked region(옷) 초기화
        #     - latent: (B, C, H, W)
        #     - mask: (B, 1, H, W)
        #     """
        #     stats = torch.load(stats_path, map_location="cpu")  # CPU에서 로드
        #     mean, std = stats["mean"].to(latent.device), stats["std"].to(latent.device)

        #     # Masked region을 저장된 통계를 기반으로 초기화
        #     noise = torch.randn_like(latent) * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

        #     # torch.where을 사용하여 마스킹된 부분만 교체
        #     latent = noise * (1. - mask) + latent * mask

        #     print("Masked region initialized.")
        #     return latent
        # C, H, W = shape
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        # start_code = initialize_masked_region(z, agnostic_mask_inversion)
        # start_code = model.q_sample(start_code, ts) 
        # mask = agnostic_mask_inversion

        """Clothing Mask Alignment and Size Adaptation W/ statistics"""
        # import torch.nn.functional as F
        # from torchvision import transforms
        # from einops import rearrange
        
        # def get_center_of_mask(mask):
        #     coords = torch.nonzero(mask[0], as_tuple=False)  # 각 배치에서 1인 좌표 찾기
        #     center = coords.float().mean(dim=0) if coords.numel() > 0 else torch.tensor([0, 0])
            
        #     return center.int()  # 모든 배치에 대한 중심 좌표를 반환


        # def pad_or_crop_to_match(image, target_shape, is_mask=True):
        #     h, w = image.shape[1:3]
        #     th, tw = target_shape
            
        #     pad_h = max(0, th - h)
        #     pad_w = max(0, tw - w)

        #     padding_value = 0 if is_mask else 0.9294
            
        #     # 패딩 적용
        #     padded_image = F.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=padding_value)
                
        #     return padded_image[..., :th, :tw]

        # def shift_to_center(image, source_mask, target_mask, is_mask=True):
        #     source_center = get_center_of_mask(source_mask)
        #     target_center = get_center_of_mask(target_mask)

        #     shift_y = target_center[0] - source_center[0]
        #     shift_x = target_center[1] - source_center[1]
            
        #     # 이미지 크기 정보
        #     height, width = image.shape[1], image.shape[2]

        #     # 이미지 텐서를 torch.roll로 이동
        #     shifted_image = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
            
        #     padding_value = 0 if is_mask else 0.9294

        #     # 롤링된 부분 삭제 (이미지 크기를 넘어선 부분 제거)
        #     if shift_y > 0:
        #         shifted_image[:, :shift_y, :] = padding_value  # 위로 이동하면 상단을 0으로 채움
        #     elif shift_y < 0:
        #         shifted_image[:, shift_y:, :] = padding_value  # 아래로 이동하면 하단을 0으로 채움

        #     if shift_x > 0:
        #         shifted_image[:, :, :shift_x] = padding_value  # 왼쪽으로 이동하면 좌측을 0으로 채움
        #     elif shift_x < 0:
        #         shifted_image[:, :, shift_x:] = padding_value  # 오른쪽으로 이동하면 우측을 0으로 채움

        #     return shifted_image


        # def align_and_overlay(inpainting_mask, clothing_mask, clothing_image):
        #     inpainting_size = torch.count_nonzero(inpainting_mask, dim=(1, 2, 3))  # 각 배치마다 마스크 크기 계산
        #     clothing_size = torch.count_nonzero(clothing_mask, dim=(1, 2, 3))
            
        #     batch_size = inpainting_mask.shape[0]
        #     resized_clothing_masks = []
        #     resized_clothing_images = []
            
        #     for i in range(batch_size):
        #         inpainting_mask_single = inpainting_mask[i]
        #         clothing_mask_single = clothing_mask[i]
        #         clothing_image_single = clothing_image[i]
                
        #         if clothing_size[i] > 0:
        #             # 크기 비율 계산 및 리사이즈
        #             scale_factor = (inpainting_size[i] / clothing_size[i]).sqrt()
        #             resize_transform = transforms.Resize((int(clothing_mask_single.shape[-2] * scale_factor),
        #                                                 int(clothing_mask_single.shape[-1] * scale_factor)))
        #             clothing_mask_resized = resize_transform(clothing_mask_single)
        #             clothing_image_resized = resize_transform(clothing_image_single)
                    
        #             # 크기 맞추기
        #             clothing_mask_resized = pad_or_crop_to_match(clothing_mask_resized, inpainting_mask_single.shape[-2:])
        #             clothing_image_resized = pad_or_crop_to_match(clothing_image_resized, inpainting_mask_single.shape[-2:], is_mask=False)
                    
        #             # 중심 맞추기
        #             clothing_mask_resized = shift_to_center(clothing_mask_resized, clothing_mask_resized, inpainting_mask_single)
        #             clothing_image_resized = shift_to_center(clothing_image_resized, clothing_image_resized, inpainting_mask_single, is_mask=False)
        #         else:
        #             clothing_mask_resized = torch.zeros_like(inpainting_mask_single)  # 빈 마스크
        #             clothing_image_resized = torch.zeros_like(inpainting_mask_single)  # 빈 이미지
                
        #         resized_clothing_masks.append(clothing_mask_resized)
        #         resized_clothing_images.append(clothing_image_resized)
            
        #     return torch.stack(resized_clothing_masks), torch.stack(resized_clothing_images)
        
        # def initialize_masked_region(latent, mask, stats_path="./masked_feature/clothing_statistics.pth"):
        #     """ 
        #     저장된 통계를 이용하여 masked region(옷) 초기화
        #     - latent: (B, C, H, W)
        #     - mask: (B, 1, H, W)
        #     """
        #     stats = torch.load(stats_path, map_location="cpu")  # CPU에서 로드
        #     mean, std = stats["mean"].to(latent.device), stats["std"].to(latent.device)

        #     # Masked region을 저장된 통계를 기반으로 초기화
        #     noise = torch.randn_like(latent) * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

        #     # torch.where을 사용하여 마스킹된 부분만 교체
        #     latent = noise * (1. - mask) + latent * mask

        #     print("Masked region initialized.")
        #     return latent
        
        # agn_mask_inversion = rearrange(batch['agn_mask'], 'b h w c -> b c h w') # raw image space
        # inpainting_mask = 1. - agn_mask_inversion
        # clothing_mask = rearrange(batch['cloth_mask'], 'b h w c -> b c h w')
        # clothing_mask, clothing_image = align_and_overlay(inpainting_mask=inpainting_mask, clothing_mask=clothing_mask, clothing_image=c["c_concat"][0])
    
        # resize_transform = transforms.Resize((64, 48))
        # resized_clothing_mask = torch.stack([resize_transform(mask) for mask in clothing_mask])
        # latent_clothing = model.get_first_stage_encoding(model.encode_first_stage(clothing_image))

        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1)  # [BS, 1, 64, 48]
        # # Unmaksed region은 z로, masked region은 statistics로 채우기기
        # start_code = initialize_masked_region(latent_clothing, resized_clothing_mask)
        # # resized_clothing_mask 영역을 latent_clothing으로 대체하기
        # start_code[agnostic_mask_inversion.repeat(1, 4, 1, 1).to(torch.bool)] = z[agnostic_mask_inversion.repeat(1, 4, 1, 1).to(torch.bool)]
        # start_code = model.q_sample(start_code, ts) 
        # mask = agnostic_mask_inversion

        """Statistics of a Clothing latent for Channel"""
        # def get_statistics(latent_clothing, clothing_mask, latent, mask):
        #     masked_clothing_features = latent_clothing * clothing_mask
            
        #     mean = masked_clothing_features.sum(dim=[2, 3]) / clothing_mask.sum(dim=[2, 3])
        #     std = ((masked_clothing_features - mean[:, :, None, None])**2).sum(dim=[2, 3]) / mask.sum(dim=[2, 3])

        #     masked_clothing_features = torch.randn_like(latent_clothing) * std.view(latent_clothing.size(0), 4, 1, 1) + mean.view((latent_clothing.size(0), 4, 1, 1))
        #     latent = masked_clothing_features * (1. - mask) + latent * mask
        #     print("Masked region initialized.")
        #     return latent
        
        # agnostic_mask_inversion = torch.cat([c["first_stage_cond"]], dim=1)[:, 4, ...].unsqueeze(1) # agnostic_mask_inversion: [BS, 1, 64, 48], unmasked region set to 1
        # latent_clothing = model.get_first_stage_encoding(model.encode_first_stage(c["c_concat"][0]))
        # from einops import rearrange
        # from torchvision import transforms
        # clothing_mask = rearrange(batch['cloth_mask'], 'b h w c -> b c h w')
        # resize_transform = transforms.Resize((64, 48))
        # resized_clothing_mask = torch.stack([resize_transform(mask) for mask in clothing_mask])
        # start_code = get_statistics(latent_clothing, resized_clothing_mask, z, agnostic_mask_inversion)
        # start_code = model.q_sample(start_code, ts) 
        # mask = agnostic_mask_inversion
        
        """-----"""

        img_callback_fn = None
        predicted_x0_fn = None
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

        if args.predicted_x0:
            def save_feature_map(feature_map, filename):
                save_feature_dir = './feature_map_999'
                os.makedirs(save_feature_dir, exist_ok=True)
                save_path = os.path.join(save_feature_dir, f"{filename}.pt")
                torch.save(feature_map, save_path)
            def predicted_x0(predx0):
                for sample_idx, (sample, fn,  cloth_fn) in enumerate(zip(predx0, batch['img_fn'], batch["cloth_fn"])):
                    save_feature_map(sample, f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}_{999}")
                x_samples = model.decode_first_stage(predx0)
                for sample_idx, (x_sample, fn,  cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
                    x_sample_img = tensor2img(x_sample)  # [0, 255]
                    to_path = opj('./feature_map_999', f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
                    cv2.imwrite(to_path, x_sample_img[:,:,::-1])
            predicted_x0_fn = predicted_x0
            
        """x_T의 잔여신호만 추출 후 feature map 추출"""
        # x_start = model.q_sample_x(start_code, ts)
        # predicted_x0(x_start)
        # breakpoint()
        
        # if args.dps: # Only use gradient method not replacement method
        #     # Add noise into measurement
        #     sigma = 0.05
        #     z = z + torch.randn_like(z, device=z.device) * sigma
        
        pixel_agnostic_mask_inversion = None
        pixel_x0 = None
        if args.pixel_mcg:
            from einops import rearrange
            pixel_agnostic_mask_inversion = rearrange(batch['agn_mask'], 'b h w c -> b c h w')
            pixel_x0 = rearrange(batch['image'], 'b h w c -> b c h w')

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
            predicted_x0=predicted_x0_fn,
            sampling_schedule=args.sampling_schedule,
            mcg=args.mcg,
            dps=args.dps,
            pixel_mcg=args.pixel_mcg,
            pixel_mask=pixel_agnostic_mask_inversion,
            pixel_x0=pixel_x0,
            noisy_mcg=args.noisy_mcg,
            unconditional_guidance_scale=args.cfg_scale,
        )

        if args.img_callback:
            continue
        
        if args.predicted_x0:
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
