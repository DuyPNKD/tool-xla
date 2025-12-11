"""
Image Quality Metrics
Tính toán các chỉ số đánh giá chất lượng ảnh
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def calculate_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Tính Mean Squared Error (MSE) giữa 2 ảnh.
    MSE càng nhỏ thì ảnh càng giống nhau.
    """
    if image1.shape != image2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước!")
    
    mse = np.mean((image1.astype(float) - image2.astype(float)) ** 2)
    return float(mse)


def calculate_psnr(image1: np.ndarray, image2: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Tính Peak Signal-to-Noise Ratio (PSNR).
    PSNR càng cao thì chất lượng ảnh càng tốt (thường > 30 dB là tốt).
    
    PSNR = 10 * log10(MAX² / MSE)
    """
    mse = calculate_mse(image1, image2)
    
    if mse == 0:
        return float('inf')  # Hai ảnh giống hệt nhau
    
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return float(psnr)


def calculate_ssim(image1: np.ndarray, image2: np.ndarray, 
                   window_size: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Tính Structural Similarity Index (SSIM).
    SSIM nằm trong khoảng [-1, 1], càng gần 1 thì ảnh càng giống.
    
    SSIM đánh giá: luminance, contrast, structure
    """
    if image1.shape != image2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước!")
    
    # Chuyển sang grayscale nếu là ảnh màu
    if len(image1.shape) == 3:
        image1_gray = (0.299 * image1[:, :, 0] + 
                      0.587 * image1[:, :, 1] + 
                      0.114 * image1[:, :, 2])
        image2_gray = (0.299 * image2[:, :, 0] + 
                      0.587 * image2[:, :, 1] + 
                      0.114 * image2[:, :, 2])
    else:
        image1_gray = image1.copy()
        image2_gray = image2.copy()
    
    image1_gray = image1_gray.astype(float)
    image2_gray = image2_gray.astype(float)
    
    # Constants
    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2
    
    # Tính mean và variance bằng Gaussian filter
    sigma = 1.5
    mu1 = gaussian_filter(image1_gray, sigma)
    mu2 = gaussian_filter(image2_gray, sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(image1_gray ** 2, sigma) - mu1_sq
    sigma2_sq = gaussian_filter(image2_gray ** 2, sigma) - mu2_sq
    sigma12 = gaussian_filter(image1_gray * image2_gray, sigma) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(np.mean(ssim_map))


def calculate_mae(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Tính Mean Absolute Error (MAE).
    """
    if image1.shape != image2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước!")
    
    mae = np.mean(np.abs(image1.astype(float) - image2.astype(float)))
    return float(mae)


def calculate_all_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """
    Tính tất cả các metrics giữa ảnh gốc và ảnh đã xử lý.
    """
    metrics = {
        'MSE': calculate_mse(original, processed),
        'PSNR': calculate_psnr(original, processed),
        'SSIM': calculate_ssim(original, processed),
        'MAE': calculate_mae(original, processed)
    }
    return metrics
