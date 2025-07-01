import albumentations as A
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2

# PSNR 계산 함수
def psnr(img1: torch.Tensor, img2: torch.Tensor, max_pixel_value: float = 255.0) -> torch.Tensor:
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value

class CustomDataset(Dataset):
    def __init__(self, original_dir, generated_dir, mask_dir, generated_img_extension: str):
        self.original_dir = original_dir
        self.generated_dir = generated_dir
        self.mask_dir = mask_dir
        
        self.original_images = os.listdir(self.original_dir)
        self.generated_images = os.listdir(self.generated_dir)
        self.mask_images = os.listdir(self.mask_dir)

        self.generated_img_extension = generated_img_extension
        
        # Albumentations의 리사이징 트랜스폼
        self.transform = A.Compose([
            A.Resize(height=512, width=384),
        ])

    def __len__(self):
        assert len(self.original_images) == len(self.generated_images), f"The number of paired datasets are not same...!(original/generated) ({len(self.original_images)}/{len(self.generated_images)})"
        return len(self.original_images)
    
    def __getitem__(self, idx):
        img_name = self.generated_images[idx]
        img_name = img_name.replace(".jpg", "")
        parts = img_name.split("_")  
        first_id = parts[0] + "_" + parts[1] 
        second_id = parts[2] + "_" + parts[3]  
        
        img_path = os.path.join(self.original_dir, first_id + ".jpg")
        gen_path = os.path.join(self.generated_dir, first_id + "_" + second_id + self.generated_img_extension)
        mask_path = os.path.join(self.mask_dir, first_id + "_mask.png")

        original = cv2.imread(img_path)
        generated = cv2.imread(gen_path)
        mask = cv2.imread(mask_path)

        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask>=128).astype(np.float32)
        
        original = self.transform(image=original)['image']
        generated = self.transform(image=generated)['image']

        mask = self.transform(image=mask)['image']
        
        # [H, W, C] → [C, H, W], 0~255 → -1~1 정규화
        original = torch.from_numpy(original).permute(2, 0, 1).float()
        generated = torch.from_numpy(generated).permute(2, 0, 1).float() 

        return {
            'original_img': original,
            'generated_img': generated,
            'mask': torch.from_numpy(mask).unsqueeze(0).float()
        }

def collate_fn(samples):
    return {
        'original_img': samples[0]['original_img'].unsqueeze(0),
        'generated_img': samples[0]['generated_img'].unsqueeze(0),
        'mask': samples[0]['mask'].unsqueeze(0)
    }

def main(original_dir, generated_dir, mask_dir, generated_img_extension):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = CustomDataset(original_dir, generated_dir, mask_dir, generated_img_extension)
    dataloader = DataLoader(dataset, num_workers=2, collate_fn=collate_fn, batch_size=1, shuffle=False)
    
    psnr_score = 0
    for batch in dataloader:
        ori, gen, mask = batch['original_img'].to(device), batch['generated_img'].to(device), batch['mask'].to(device)

        mask = (mask > 0.5).float()
        mask = (1.0 - mask)  # invert to get visible region

        ori_masked = ori * mask
        gen_masked = gen * mask

        psnr_score += psnr(ori_masked, gen_masked, max_pixel_value=255.0)
        
    return psnr_score / len(dataloader)

if __name__ == "__main__":
    original_dir = '/home/qkrwnstj/StableVITON/dataset/zalando-hd-resized/test/image'
    mask_dir = '/home/qkrwnstj/StableVITON/dataset/zalando-hd-resized/test/agnostic-mask'
    generated_img_extension = '.jpg'

    dir_name = "999_ddpmPrior_woNoise_cfg1"
    generated_dir_dict = {
        'pair': f'/home/qkrwnstj/StableVITON/{dir_name}/pair',
        'unpair': f'/home/qkrwnstj/StableVITON/{dir_name}/unpair'
    }

    total_score = 0
    for name, gen_dir in generated_dir_dict.items():
        PSNR_score = main(original_dir, gen_dir, mask_dir, generated_img_extension)
        print(f"[{name}] Masked PSNR score: {PSNR_score:.4f}")
        total_score += PSNR_score
    
    print(f"[Total] Masked PSNR score: {total_score / 2:.4f}")