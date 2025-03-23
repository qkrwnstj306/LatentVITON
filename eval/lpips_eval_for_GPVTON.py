import lpips
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, original_dir, generated_dir, generated_img_extension: str, target: str):
        self.original_dir = original_dir
        self.generated_dir = generated_dir
        self.target = target
        
        self.original_images = os.listdir(self.original_dir)
        self.generated_images = os.listdir(self.generated_dir)

        self.generated_img_extension = generated_img_extension
        
        self.transform = transforms.Compose([
            transforms.Resize((512,384), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.original_images)
    
    def __getitem__(self, idx):
        img_name = self.original_images[idx]
        img_path = os.path.join(self.original_dir, img_name)
        
        img_name_without_extension = img_name.replace(".jpg", "")
        
        if self.target == 'gp-vton':
            generated_img_path = os.path.join(self.generated_dir, "upper___" + img_name_without_extension + self.generated_img_extension + "___" + img_name_without_extension + self.generated_img_extension)

        elif self.target == 'ladi-vton' or self.target == 'dci-vton' or self.target == 'dci-vton2':
            generated_img_path = os.path.join(self.generated_dir, img_name_without_extension + self.generated_img_extension)
        
        original_img = Image.open(img_path).convert('RGB')
        
        try:
            generated_img = Image.open(generated_img_path).convert('RGB')
        except FileNotFoundError:
            print(generated_img_path)
            print(f"Generated image not found for {img_name}. Skipping...")
            return None
        
        original_img, generated_img = self.transform(original_img), self.transform(generated_img)
        
        return {'original_img':original_img, 'generated_img':generated_img}

def collate_fn(samples):
    if samples[0] is None:  # Skip iteration if samples[0] is None
        return None
    original_img = samples[0]['original_img'].unsqueeze(0)
    generated_img = samples[0]['generated_img'].unsqueeze(0)
    return {'original_img':original_img, 'generated_img':generated_img}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_dir = './HR-VITON/data/zalando-hd-resized/test/image'
    
    generated_dir_lst = {'gp-vton': './GP-VTON/sample/test_gpvtongen_vitonhd_paired_1109',
                         'ladi-vton': './ladi-vton/results/paired/upper_body',
                         'dci-vton': '../xai/DCI-VTON-Virtual-Try-On/results/viton/result/paired',
                         'dci-vton2': '../xai/DCI-VTON-Virtual-Try-On/results/viton/uninversion/paired'
                         }
    
    target = 'gp-vton'
    generated_dir = generated_dir_lst[target]
    generated_img_extension = '.png'

    dataset = CustomDataset(original_dir, generated_dir, generated_img_extension, target)

    dataloader = DataLoader(dataset, num_workers=2, collate_fn=collate_fn, batch_size=1, shuffle=False)

    loss_fn = lpips.LPIPS(net='alex').to(device)
    score = 0
    
    for idx, batch in enumerate(dataloader):
        if batch is None:  # Skip iteration if batch is None
            continue
        original_img, generated_img = batch['original_img'].to(device), batch['generated_img'].to(device)
        score += loss_fn(original_img, generated_img)
        
    return score / len(dataloader)
    
if __name__ == "__main__":
    LPIPS_score = main()
    print(f"LPIPS_score: {LPIPS_score}")
