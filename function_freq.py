import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.fft
import pywt  # pip install PyWavelets
import numpy as np

"""
Spatial domain
1. Laplacian filter: 고주파 성분 = 국지적인 변화 (즉, edge, texture)
2. Difference of Gaussians (DoG): 두 개의 Gaussian blurred 이미지 차이로 고주파 강조:
"""

"""
DoG -> replaced_latent = replace_high_frequency(other_latent, pred_x0)
"""
def apply_gaussian_blur(x, kernel_size=5, sigma=1.0):
    """
    채널별로 gaussian blur 적용 (torchvision은 1채널 또는 3채널만 직접 지원)
    """
    B, C, H, W = x.shape
    x_blurred = []
    for c in range(C):
        # 각 채널을 blur 처리 후 다시 쌓기
        channel = x[:, c:c+1, :, :]
        blurred = torch.stack([
            TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=sigma)
            for img in channel
        ])
        x_blurred.append(blurred)
    return torch.cat(x_blurred, dim=1)

def extract_high_frequency_dog(x, sigma1=1.0, sigma2=2.0):
    blur1 = apply_gaussian_blur(x, kernel_size=5, sigma=sigma1)
    blur2 = apply_gaussian_blur(x, kernel_size=5, sigma=sigma2)
    return blur1 - blur2

def remove_high_frequency_dog(x, sigma=2.0):
    return apply_gaussian_blur(x, kernel_size=5, sigma=sigma)

def replace_high_frequency_dog(target_latent, source_latent, sigma=2.0):
    target_low = remove_high_frequency_dog(target_latent, sigma=sigma)
    source_high = extract_high_frequency_dog(source_latent, sigma1=1.0, sigma2=sigma)
    return target_low + source_high

"""
Laplacian
"""
def extract_high_frequency_laplacian(x):
    # Laplacian 필터 커널 정의
    lap_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]
    lap_kernel = lap_kernel.repeat(x.size(1), 1, 1, 1)  # [C,1,3,3]

    # 채널별 convolution
    high = F.conv2d(x, lap_kernel, padding=1, groups=x.size(1))
    return high

def remove_high_frequency_laplacian(x):
    high = extract_high_frequency_laplacian(x)
    return x - high

def replace_high_frequency_laplacian(target_latent, source_latent):
    target_low = remove_high_frequency_laplacian(target_latent)
    source_high = extract_high_frequency_laplacian(source_latent)
    return target_low + source_high

"""
Fourier domain
2D FFT (Fourier Transform): 이미지나 latent map에 대해 2D FFT를 수행하고, 고주파 영역만 남기거나 분리
"""
def extract_high_frequency_fft(x, keep_ratio=0.1):
    B, C, H, W = x.shape
    x_fft = torch.fft.fft2(x)
    x_fftshift = torch.fft.fftshift(x_fft)

    # 중앙 고주파 영역 제외
    center_h, center_w = H // 2, W // 2
    margin_h, margin_w = int(H * keep_ratio // 2), int(W * keep_ratio // 2)
    mask = torch.ones_like(x_fftshift)
    mask[:, :, center_h-margin_h:center_h+margin_h, center_w-margin_w:center_w+margin_w] = 0

    high_fft = x_fftshift * mask
    high = torch.fft.ifft2(torch.fft.ifftshift(high_fft)).real
    return high

def remove_high_frequency_fft(x, keep_ratio=0.1):
    B, C, H, W = x.shape
    x_fft = torch.fft.fft2(x)
    x_fftshift = torch.fft.fftshift(x_fft)

    center_h, center_w = H // 2, W // 2
    margin_h, margin_w = int(H * keep_ratio // 2), int(W * keep_ratio // 2)
    mask = torch.zeros_like(x_fftshift)
    mask[:, :, center_h-margin_h:center_h+margin_h, center_w-margin_w:center_w+margin_w] = 1

    low_fft = x_fftshift * mask
    low = torch.fft.ifft2(torch.fft.ifftshift(low_fft)).real
    return low

def replace_high_frequency_fft(target_latent, source_latent, keep_ratio=0.1):
    low = remove_high_frequency_fft(target_latent, keep_ratio)
    high = extract_high_frequency_fft(source_latent, keep_ratio)
    return low + high
