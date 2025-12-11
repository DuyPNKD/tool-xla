"""
Custom Image Processing Algorithms
Tự implement các thuật toán xử lý ảnh từ đầu không dùng OpenCV
"""

import numpy as np


# ==================== CONVOLUTION & FILTERING ====================

def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Tạo Gaussian kernel từ công thức toán học.
    G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    # Tính tổng để normalize
    sum_val = 0.0
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum_val += kernel[i, j]
    
    # Normalize kernel
    kernel /= sum_val
    return kernel


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Tự implement convolution 2D.
    """
    if len(image.shape) == 3:
        # Xử lý từng channel riêng
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = convolve2d(image[:, :, c], kernel)
        return result
    
    # Padding để giữ kích thước output
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='edge')
    
    h, w = image.shape
    kh, kw = kernel.shape
    result = np.zeros_like(image, dtype=np.float64)
    
    # Convolution
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def custom_gaussian_blur(image: np.ndarray, ksize: int, sigma: float = None) -> np.ndarray:
    """
    Tự implement Gaussian Blur từ đầu.
    """
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    kernel = create_gaussian_kernel(ksize, sigma)
    return convolve2d(image, kernel)


def custom_median_filter(image: np.ndarray, ksize: int) -> np.ndarray:
    """
    Tự implement Median Filter.
    Lấy giá trị median trong cửa sổ ksize x ksize.
    """
    if len(image.shape) == 3:
        result = np.zeros_like(image)
        for c in range(image.shape[2]):
            result[:, :, c] = custom_median_filter(image[:, :, c], ksize)
        return result
    
    pad = ksize // 2
    padded = np.pad(image, pad, mode='edge')
    
    h, w = image.shape
    result = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+ksize, j:j+ksize]
            result[i, j] = np.median(region)
    
    return result.astype(np.uint8)


# ==================== EDGE DETECTION ====================

def custom_sobel_operator(image: np.ndarray, direction: str = 'both') -> np.ndarray:
    """
    Tự implement Sobel operator để phát hiện biên.
    """
    # Chuyển sang grayscale nếu là ảnh màu
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float64)
    
    # Áp dụng convolution
    if direction == 'x':
        gradient = convolve2d(gray, sobel_x)
    elif direction == 'y':
        gradient = convolve2d(gray, sobel_y)
    else:  # both - magnitude
        gx = convolve2d(gray.astype(np.float64), sobel_x)
        gy = convolve2d(gray.astype(np.float64), sobel_y)
        gradient = np.sqrt(gx**2 + gy**2)
    
    # Normalize về [0, 255]
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    
    # Chuyển về 3 channels nếu cần
    if len(image.shape) == 3:
        gradient = np.stack([gradient] * 3, axis=-1)
    
    return gradient


def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Non-maximum suppression cho Canny edge detection.
    """
    h, w = magnitude.shape
    result = np.zeros_like(magnitude)
    
    # Làm tròn góc về 4 hướng: 0, 45, 90, 135
    angle = np.rad2deg(angle) % 180
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            
            # Xác định 2 pixel láng giềng theo hướng gradient
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= angle[i,j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif 67.5 <= angle[i,j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif 112.5 <= angle[i,j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            
            # Giữ lại nếu là cực đại cục bộ
            if magnitude[i,j] >= q and magnitude[i,j] >= r:
                result[i,j] = magnitude[i,j]
            else:
                result[i,j] = 0
    
    return result


def double_threshold(image: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15) -> tuple:
    """
    Double threshold cho Canny edge detection.
    """
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio
    
    strong = 255
    weak = 75
    
    result = np.zeros_like(image)
    
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    return result, weak, strong


def edge_tracking_by_hysteresis(image: np.ndarray, weak: int, strong: int) -> np.ndarray:
    """
    Edge tracking bằng hysteresis cho Canny.
    """
    h, w = image.shape
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if image[i, j] == weak:
                # Kiểm tra 8 láng giềng
                if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or 
                    (image[i+1, j+1] == strong) or (image[i, j-1] == strong) or 
                    (image[i, j+1] == strong) or (image[i-1, j-1] == strong) or 
                    (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                    image[i, j] = strong
                else:
                    image[i, j] = 0
    
    return image


def custom_canny_edge(image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Tự implement Canny Edge Detection từ đầu.
    Các bước:
    1. Gaussian blur để giảm nhiễu
    2. Tính gradient (Sobel)
    3. Non-maximum suppression
    4. Double threshold
    5. Edge tracking by hysteresis
    """
    # Chuyển sang grayscale
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    # Bước 1: Gaussian blur
    blurred = custom_gaussian_blur(gray, 5, 1.4)
    
    # Bước 2: Tính gradient với Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    gx = convolve2d(blurred.astype(np.float64), sobel_x)
    gy = convolve2d(blurred.astype(np.float64), sobel_y)
    
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    
    # Bước 3: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, angle)
    
    # Bước 4 & 5: Double threshold và hysteresis
    thresholded, weak, strong = double_threshold(suppressed, low_threshold/255, high_threshold/255)
    edges = edge_tracking_by_hysteresis(thresholded.copy(), weak, strong)
    
    # Chuyển về 3 channels
    if len(image.shape) == 3:
        edges = np.stack([edges] * 3, axis=-1)
    
    return edges.astype(np.uint8)


# ==================== THRESHOLDING ====================

def custom_global_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    """
    Tự implement Global Thresholding.
    """
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    result = np.where(gray > threshold, 255, 0).astype(np.uint8)
    
    if len(image.shape) == 3:
        result = np.stack([result] * 3, axis=-1)
    
    return result


def custom_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Tự implement Otsu's thresholding.
    Tìm ngưỡng tối ưu bằng cách maximize between-class variance.
    """
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    # Tính histogram
    hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(float)
    
    # Normalize histogram
    hist /= hist.sum()
    
    # Tìm ngưỡng tối ưu
    max_variance = 0
    optimal_threshold = 0
    
    for t in range(1, 256):
        # Xác suất class 0 (background)
        w0 = hist[:t].sum()
        if w0 == 0:
            continue
        
        # Xác suất class 1 (foreground)
        w1 = hist[t:].sum()
        if w1 == 0:
            break
        
        # Mean của class 0
        mu0 = (np.arange(t) * hist[:t]).sum() / w0
        
        # Mean của class 1
        mu1 = (np.arange(t, 256) * hist[t:]).sum() / w1
        
        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t
    
    # Áp dụng threshold
    result = np.where(gray > optimal_threshold, 255, 0).astype(np.uint8)
    
    if len(image.shape) == 3:
        result = np.stack([result] * 3, axis=-1)
    
    return result


def custom_adaptive_threshold(image: np.ndarray, block_size: int = 11, C: int = 2, method: str = 'mean') -> np.ndarray:
    """
    Tự implement Adaptive Thresholding.
    """
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    result = np.zeros_like(gray)
    pad = block_size // 2
    padded = np.pad(gray, pad, mode='edge')
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+block_size, j:j+block_size]
            
            if method == 'mean':
                threshold = region.mean() - C
            else:  # gaussian
                # Tạo gaussian weight
                kernel = create_gaussian_kernel(block_size, block_size / 6)
                threshold = np.sum(region * kernel) / kernel.sum() - C
            
            result[i, j] = 255 if gray[i, j] > threshold else 0
    
    if len(image.shape) == 3:
        result = np.stack([result] * 3, axis=-1)
    
    return result.astype(np.uint8)


# ==================== MORPHOLOGICAL OPERATIONS ====================

def custom_erosion(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Tự implement Erosion.
    Lấy giá trị minimum trong cửa sổ kernel.
    """
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    pad = kernel_size // 2
    padded = np.pad(gray, pad, mode='edge')
    
    h, w = gray.shape
    result = np.zeros_like(gray)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = region.min()
    
    if len(image.shape) == 3:
        result = np.stack([result] * 3, axis=-1)
    
    return result.astype(np.uint8)


def custom_dilation(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Tự implement Dilation.
    Lấy giá trị maximum trong cửa sổ kernel.
    """
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    pad = kernel_size // 2
    padded = np.pad(gray, pad, mode='edge')
    
    h, w = gray.shape
    result = np.zeros_like(gray)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = region.max()
    
    if len(image.shape) == 3:
        result = np.stack([result] * 3, axis=-1)
    
    return result.astype(np.uint8)


def custom_opening(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Tự implement Opening = Erosion -> Dilation.
    Loại bỏ nhiễu nhỏ.
    """
    eroded = custom_erosion(image, kernel_size)
    opened = custom_dilation(eroded, kernel_size)
    return opened


def custom_closing(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Tự implement Closing = Dilation -> Erosion.
    Lấp đầy các lỗ nhỏ.
    """
    dilated = custom_dilation(image, kernel_size)
    closed = custom_erosion(dilated, kernel_size)
    return closed


# ==================== HISTOGRAM EQUALIZATION ====================

def custom_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Tự implement Histogram Equalization.
    Cân bằng histogram để tăng độ tương phản.
    """
    if len(image.shape) == 3:
        # Xử lý từng channel
        result = np.zeros_like(image)
        for c in range(3):
            result[:, :, c] = custom_histogram_equalization(image[:, :, c])
        return result
    
    # Tính histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Tính CDF (Cumulative Distribution Function)
    cdf = hist.cumsum()
    
    # Normalize CDF về [0, 255]
    cdf_normalized = 255 * cdf / cdf[-1]
    
    # Map giá trị pixel cũ sang giá trị mới
    result = np.interp(image.flatten(), np.arange(256), cdf_normalized)
    result = result.reshape(image.shape).astype(np.uint8)
    
    return result


# ==================== SHARPEN ====================

def custom_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Tự implement Sharpening filter.
    Sử dụng Laplacian kernel để tăng cường biên.
    """
    # Laplacian kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float64)
    
    result = convolve2d(image, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


# ==================== UTILITY FUNCTIONS ====================

def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển RGB sang grayscale theo công thức:
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if len(image.shape) == 2:
        return image
    
    return (0.299 * image[:, :, 0] + 
            0.587 * image[:, :, 1] + 
            0.114 * image[:, :, 2]).astype(np.uint8)
