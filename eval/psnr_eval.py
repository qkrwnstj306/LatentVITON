import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

def psnr(img1: torch.Tensor, img2: torch.Tensor, max_pixel_value: float = 255.0) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        img1 (torch.Tensor): First image tensor of shape (N, C, H, W).
        img2 (torch.Tensor): Second image tensor of the same shape as img1.
        max_pixel_value (float): Maximum pixel value in the images (default: 255.0 for 8-bit images).
    
    Returns:
        torch.Tensor: PSNR value.
    """
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return torch.tensor(float('inf'))
    psnr_value = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr_value

class CustomDataset(Dataset):
    def __init__(self, original_dir, generated_dir, generated_img_extension: str):
        self.original_dir = original_dir
        self.generated_dir = generated_dir
        
        self.original_images = os.listdir(self.original_dir)
        self.generated_images = os.listdir(self.generated_dir)

        self.generated_img_extension = generated_img_extension
        
        self.transform = transforms.Compose([
            transforms.Resize((512,384), Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        assert len(self.original_images) == len(self.generated_images), f"The number of paired datasets are not same...!(original/generated) ({len(self.original_images)/len(self.generated_images)})"
        return len(self.original_images)
    
    def __getitem__(self, idx):
        img_name = self.original_images[idx]
        img_path = os.path.join(self.original_dir, img_name)
        
        img_name_without_extension = img_name.replace(".jpg", "")
        generated_img_path = os.path.join(self.generated_dir,img_name_without_extension + "_" + img_name_without_extension + self.generated_img_extension)
        
        original_img = Image.open(img_path).convert('RGB')
        generated_img = Image.open(generated_img_path).convert('RGB')
        
        original_img, generated_img = self.transform(original_img), self.transform(generated_img)
        
        return {'original_img':original_img, 'generated_img':generated_img}

def collate_fn(samples):
    original_img = samples[0]['original_img'].unsqueeze(0)
    generated_img = samples[0]['generated_img'].unsqueeze(0)
    return {'original_img':original_img, 'generated_img':generated_img}

def main(original_dir, generated_dir, generated_img_extension):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_dir = original_dir
    generated_dir = generated_dir
    generated_img_extension = generated_img_extension

    dataset = CustomDataset(original_dir, generated_dir, generated_img_extension)
    dataloader = DataLoader(dataset, num_workers=2, collate_fn=collate_fn, batch_size=1, shuffle=False)
    
    psnr_score = 0
    for idx, batch in enumerate(dataloader):
        original_img, generated_img = batch['original_img'].to(device), batch['generated_img'].to(device)
        original_img, generated_img = original_img * 255. , generated_img * 255.
        
        psnr_score += psnr(original_img, generated_img, max_pixel_value=255.0)
        
    return psnr_score / len(dataloader)
    
if __name__ == "__main__":
    original_dir = '/home/qkrwnstj/StableVITON/dataset/zalando-hd-resized/test/image'
    
    generated_dir_lst = ['/home/qkrwnstj/StableVITON/stableVITON/pair', 
                         '/home/qkrwnstj/StableVITON/ours/pair',
                         '/home/qkrwnstj/StableVITON/original/pair',
                         '/home/qkrwnstj/StableVITON/ours_latents/pair',
                         '/home/qkrwnstj/StableVITON/s_l_VITON/pair',
                         '/home/qkrwnstj/StableVITON/s_l_VITON_repaint/pair',
                         '/home/qkrwnstj/StableVITON/999_unmasked_replacement_cfg_3/pair',
                         ]
    
    generated_dir = generated_dir_lst[6]
    
    generated_img_extension = '.jpg'
    
    PSNR_score = main(original_dir, generated_dir, generated_img_extension)
    print(f"PSNR_score: {PSNR_score}")