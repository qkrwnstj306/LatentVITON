import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import albumentations as A
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, original_dir, generated_dir, mask_dir, generated_img_extension: str):
        self.original_dir = original_dir
        self.generated_dir = generated_dir
        self.mask_dir = mask_dir
        
        self.original_images = os.listdir(self.original_dir)
        self.generated_images = os.listdir(self.generated_dir)
        self.mask_images = os.listdir(self.mask_dir)

        self.generated_img_extension = generated_img_extension
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
        mask = cv2.imread(mask_path)  # assuming single-channel mask
        
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

def masked_mse(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute MSE only over the unmasked regions.
    """
    diff = (img1 - img2) ** 2
    masked_diff = diff * mask
    return masked_diff.sum() / mask.sum()

def main(original_dir, generated_dir, mask_dir, generated_img_extension):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = CustomDataset(original_dir, generated_dir, mask_dir, generated_img_extension)
    dataloader = DataLoader(dataset, num_workers=2, collate_fn=collate_fn, batch_size=1, shuffle=False)
    
    total_mse = 0
    for batch in dataloader:
        ori = batch['original_img'].to(device)
        gen = batch['generated_img'].to(device)
        mask = batch['mask'].to(device)

        # Normalize mask to binary (0 or 1), then invert
        mask = (mask > 0.5).float()
        mask = 1.0 - mask

        mse_value = masked_mse(ori, gen, mask)
        total_mse += mse_value

    return total_mse / len(dataloader)

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
        MSE_score = main(original_dir, gen_dir, mask_dir, generated_img_extension)
        print(f"[{name}] Masked MSE score: {MSE_score:.4f}")
        total_score += MSE_score
    
    print(f"[Total] Masked MSE score: {total_score / 2:.4f}")
