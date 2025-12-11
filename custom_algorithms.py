"""
==================================================================================
                    CUSTOM IMAGE PROCESSING ALGORITHMS
==================================================================================

Tập hợp các thuật toán xử lý ảnh được tự implement từ đầu (không dùng OpenCV).
Tất cả thuật toán chỉ sử dụng NumPy cho tính toán ma trận.

📦 MODULE STRUCTURE:
    1. CONVOLUTION & FILTERING - Lọc và làm mịn ảnh
    2. EDGE DETECTION - Phát hiện biên và gradient
    3. THRESHOLDING - Phân ngưỡng và phân đoạn ảnh
    4. MORPHOLOGICAL OPERATIONS - Xử lý hình thái học
    5. ENHANCEMENT - Tăng cường chất lượng ảnh
    6. UTILITY FUNCTIONS - Các hàm tiện ích

🎯 FEATURES:
    ✅ 11 thuật toán xử lý ảnh chính
    ✅ Hỗ trợ cả ảnh grayscale và RGB
    ✅ Tự động xử lý padding và normalization
    ✅ Code tối ưu với NumPy vectorization

👨‍💻 AUTHOR: DuyPNKD
📅 DATE: 2025
📚 COURSE: Image Processing - Semester 6

==================================================================================
"""

import numpy as np


# ==================== CONVOLUTION & FILTERING ====================

def create_gaussian_kernel(size: int, sigma: float = 1.0) -> np.ndarray:
    """
    Tạo Gaussian kernel 2D từ công thức toán học.
    
    🔢 CÔNG THỨC:
        G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
    
    🎯 MỤC ĐÍCH:
        Tạo ma trận Gaussian để làm mờ ảnh theo phân phối chuẩn.
        Pixel ở trung tâm có trọng số cao nhất, giảm dần về các cạnh.
    
    📊 THAM SỐ:
        size (int): Kích thước kernel (3, 5, 7, 9, ...)
        sigma (float): Độ lệch chuẩn - Càng lớn = mờ càng mạnh
    
    ✨ TÍNH NĂNG:
        - Tự động normalize tổng = 1
        - Kernel đối xứng qua tâm
        - Phù hợp cho Gaussian Blur và Canny Edge Detection
    
    💡 VÍ DỤ:
        >>> kernel = create_gaussian_kernel(5, 1.0)
        >>> kernel.shape  # (5, 5)
        >>> kernel.sum()  # ~1.0
    
    📖 THAM KHẢO:
        - Gonzalez & Woods: Digital Image Processing, Chapter 3
        - https://en.wikipedia.org/wiki/Gaussian_blur
    """
    # Validation
    if size < 3:
        raise ValueError(f"Kernel size phải ≥ 3, nhận được: {size}")
    if size % 2 == 0:
        raise ValueError(f"Kernel size phải là số lẻ, nhận được: {size}")
    if sigma <= 0:
        raise ValueError(f"Sigma phải > 0, nhận được: {sigma}")
    
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
    Tự implement Convolution 2D - Thao tác cơ bản nhất trong xử lý ảnh.
    
    🔄 NGUYÊN LÝ:
        Trượt kernel qua từng vị trí trên ảnh, tính tổng tích element-wise.
        Output[i,j] = Σ Σ Image[i+m,j+n] * Kernel[m,n]
    
    🎯 MỤC ĐÍCH:
        - Áp dụng filter (blur, sharpen, edge detection)
        - Cơ sở cho hầu hết các thuật toán xử lý ảnh
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào (H×W hoặc H×W×C)
        kernel (np.ndarray): Ma trận filter (thường 3×3, 5×5, 7×7)
    
    ✨ TÍNH NĂNG:
        - Tự động xử lý ảnh RGB (convolution từng channel riêng)
        - Edge padding để giữ nguyên kích thước output
        - Clipping về [0, 255] để tránh overflow
    
    ⚡ HIỆU SUẤT:
        - O(H × W × K²) với K là kernel size
        - Có thể tối ưu bằng FFT cho kernel lớn (chưa implement)
    
    💡 VÍ DỤ:
        >>> blur_kernel = create_gaussian_kernel(5)
        >>> blurred = convolve2d(image, blur_kernel)
    """
    # Validation
    if image.size == 0:
        raise ValueError("Ảnh rỗng (empty image)")
    if kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"Kernel phải là ma trận vuông, nhận: {kernel.shape}")
    if kernel.shape[0] % 2 == 0:
        raise ValueError(f"Kernel size phải lẻ, nhận: {kernel.shape[0]}")
    
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
    
    # Convolution với tối ưu
    # TODO: Có thể tối ưu thêm bằng vectorization hoặc scipy.signal.convolve2d
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    
    return np.clip(result, 0, 255).astype(np.uint8)


# Cache cho Gaussian kernels (tăng hiệu suất)
_gaussian_kernel_cache = {}

def custom_gaussian_blur(image: np.ndarray, ksize: int, sigma: float = None) -> np.ndarray:
    """
    🌫️ GAUSSIAN BLUR - Làm mờ ảnh theo phân phối Gaussian
    
    🎯 MỤC ĐÍCH:
        Làm mịn ảnh để:
        - Giảm nhiễu (noise reduction)
        - Chuẩn bị cho edge detection
        - Tạo hiệu ứng bokeh/depth-of-field
    
    🔬 THUẬT TOÁN:
        1. Tạo Gaussian kernel với sigma cho trước
        2. Áp dụng convolution 2D lên ảnh
        3. Normalize kết quả về [0, 255]
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào (H×W hoặc H×W×3)
        ksize (int): Kích thước kernel (3, 5, 7, 9, ...)
        sigma (float, optional): Độ lệch chuẩn
            - None: tự động tính theo công thức OpenCV
            - 0.5-1.0: mờ nhẹ
            - 1.0-2.0: mờ vừa
            - >2.0: mờ mạnh
    
    ✅ ƯU ĐIỂM so với Average Blur:
        - Giữ biên tốt hơn (trọng số giảm dần từ tâm)
        - Tự nhiên hơn cho mắt người
        - Ít tạo artifact
    
    ⚠️ CHÚ Ý:
        - Kernel size càng lớn → chậm hơn
        - Không phù hợp cho nhiễu "muối tiêu" (dùng Median thay thế)
    
    💡 ỨNG DỤNG:
        - Tiền xử lý cho Canny Edge Detection
        - Làm mờ background trong chụp ảnh chân dung
        - Giảm nhiễu trong ảnh y khoa
    
    📖 THAM KHẢO:
        - Gonzalez & Woods, Chapter 3.4: Smoothing Spatial Filters
    """
    if sigma is None:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    
    # Sử dụng cache để tránh tính lại kernel giống nhau
    cache_key = (ksize, round(sigma, 3))
    if cache_key not in _gaussian_kernel_cache:
        _gaussian_kernel_cache[cache_key] = create_gaussian_kernel(ksize, sigma)
    kernel = _gaussian_kernel_cache[cache_key]
    
    return convolve2d(image, kernel)


def custom_median_filter(image: np.ndarray, ksize: int) -> np.ndarray:
    """
    🔢 MEDIAN FILTER - Lọc trung vị để khử nhiễu
    
    🎯 MỤC ĐÍCH:
        Loại bỏ nhiễu "muối tiêu" (salt-and-pepper noise) hiệu quả:
        - Giữ nguyên biên sắc nét
        - Không làm mờ ảnh như Gaussian
        - Thay pixel bằng giá trị median của vùng lân cận
    
    🔬 THUẬT TOÁN:
        1. Duyệt qua từng pixel
        2. Lấy cửa sổ ksize×ksize xung quanh
        3. Sắp xếp các giá trị và lấy median
        4. Thay thế pixel trung tâm = median
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào
        ksize (int): Kích thước cửa sổ (3, 5, 7)
            - 3: nhanh, khử nhiễu nhẹ
            - 5: cân bằng tốc độ và hiệu quả
            - 7+: chậm, khử nhiễu mạnh
    
    ✅ ƯU ĐIỂM:
        - Rất hiệu quả với nhiễu muối tiêu
        - Giữ biên tốt (non-linear filter)
        - Không tạo blur như Gaussian
    
    ❌ NHƯỢC ĐIỂM:
        - Chậm hơn linear filters (O(n log n) do sorting)
        - Có thể làm mất chi tiết nhỏ
        - Không hiệu quả với nhiễu Gaussian
    
    💡 ỨNG DỤNG:
        - Khử nhiễu trong ảnh scan, photocopy
        - Xử lý ảnh vệ tinh
        - Tiền xử lý cho OCR
    
    🔬 SO SÁNH:
        vs Gaussian Blur: Giữ biên tốt hơn
        vs Mean Filter: Ít nhạy cảm với outliers
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
    🔍 SOBEL OPERATOR - Phát hiện biên bằng gradient
    
    🎯 MỤC ĐÍCH:
        Tìm biên trong ảnh bằng cách tính gradient theo hướng X và Y:
        - Phát hiện thay đổi cường độ sáng đột ngột
        - Xác định hướng và độ mạnh của biên
    
    🔬 THUẬT TOÁN:
        1. Chuyển ảnh về grayscale
        2. Áp dụng Sobel kernels:
            Gx = [[-1, 0, 1],     Gy = [[-1, -2, -1],
                  [-2, 0, 2],           [ 0,  0,  0],
                  [-1, 0, 1]]           [ 1,  2,  1]]
        3. Tính magnitude: G = √(Gx² + Gy²)
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào
        direction (str): Hướng gradient
            - 'x': Phát hiện biên dọc (vertical edges)
            - 'y': Phát hiện biên ngang (horizontal edges)
            - 'both': Magnitude tổng hợp (mặc định)
    
    🔢 CÔNG THỨC:
        - Gx: Gradient theo chiều ngang (thay đổi trái-phải)
        - Gy: Gradient theo chiều dọc (thay đổi trên-dưới)
        - Magnitude: √(Gx² + Gy²)
        - Direction: arctan(Gy/Gx)
    
    ✅ ƯU ĐIỂM:
        - Đơn giản, nhanh
        - Giảm nhiễu tốt (có smoothing)
        - Phát hiện cả hướng biên
    
    ❌ NHƯỢC ĐIỂM:
        - Nhạy cảm với nhiễu
        - Biên dày (thick edges)
        - Không optimal như Canny
    
    💡 ỨNG DỤNG:
        - Tiền xử lý cho object detection
        - Tìm đường viền trong CAD
        - Phân tích kết cấu (texture)
    
    📖 THAM KHẢO:
        - Sobel, I. (1968): "An Isotropic 3×3 Image Gradient Operator"
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
    ⭐ CANNY EDGE DETECTION - Thuật toán phát hiện biên tối ưu
    
    🏆 ĐẶC ĐIỂM:
        Thuật toán phát hiện biên TỐT NHẤT được phát triển bởi John Canny (1986):
        - Biên mỏng (single-pixel)
        - Ít nhiễu giả (false edges)
        - Chính xác cao
    
    🔬 5 BƯỚC THỰC HIỆN:
        
        BƯỚC 1️⃣: NOISE REDUCTION (Giảm nhiễu)
            → Áp dụng Gaussian Blur (5×5, σ=1.4)
            → Lý do: Canny rất nhạy nhiễu
        
        BƯỚC 2️⃣: GRADIENT CALCULATION (Tính gradient)
            → Sobel operators: Gx, Gy
            → Magnitude: G = √(Gx² + Gy²)
            → Direction: θ = arctan(Gy/Gx)
        
        BƯỚC 3️⃣: NON-MAXIMUM SUPPRESSION (Làm mỏng biên)
            → Giữ lại pixel cực đại theo hướng gradient
            → Loại bỏ pixel không phải biên chính
            → Kết quả: biên dày 1 pixel
        
        BƯỚC 4️⃣: DOUBLE THRESHOLD (Ngưỡng kép)
            → Strong edges: G > high_threshold
            → Weak edges: low_threshold < G < high_threshold
            → Non-edges: G < low_threshold
        
        BƯỚC 5️⃣: EDGE TRACKING BY HYSTERESIS (Theo dõi biên)
            → Giữ weak edges nếu kết nối với strong edges
            → Loại bỏ weak edges đứng riêng
            → Kết quả: biên liên tục
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào
        low_threshold (int): Ngưỡng thấp (50-100)
            - Thấp: nhiều biên, nhiều nhiễu
            - Cao: ít biên, mất chi tiết
        high_threshold (int): Ngưỡng cao (150-200)
            - Tỷ lệ khuyến nghị: high = 2-3 × low
    
    ⚙️ CÁCH CHỌN THRESHOLD:
        - Ảnh nhiễu nhiều: Tăng cả 2 ngưỡng
        - Muốn nhiều chi tiết: Giảm cả 2
        - Tỷ lệ high/low = 2:1 hoặc 3:1
    
    ✅ ƯU ĐIỂM:
        - Biên mỏng, chính xác nhất
        - Ít nhiễu giả (false positives)
        - Biên liên tục
    
    ❌ NHƯỢC ĐIỂM:
        - Chậm (5 bước xử lý)
        - Cần điều chỉnh threshold
        - Không phát hiện góc tốt
    
    💡 ỨNG DỤNG:
        - Computer Vision cơ bản
        - Lane detection (xe tự lái)
        - Medical imaging
        - Object recognition
    
    📖 THAM KHẢO:
        - Canny, J. (1986): "A Computational Approach to Edge Detection"
        - IEEE TPAMI, Vol. PAMI-8, No. 6
    
    🎯 TẠI SAO CANNY TỐT NHẤT?
        1. Good detection: Tìm đúng biên thật
        2. Good localization: Biên đúng vị trí
        3. Single response: Mỗi biên chỉ 1 pixel
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
    🎯 OTSU'S THRESHOLDING - Tự động tìm ngưỡng tối ưu
    
    🏆 ĐẶC ĐIỂM:
        Thuật toán tự động tìm ngưỡng phân đoạn TỐI ƯU (Nobuyuki Otsu, 1979):
        - Không cần tham số đầu vào
        - Tối ưu hóa toán học
        - Phù hợp cho ảnh bimodal histogram
    
    🔬 THUẬT TOÁN:
        
        1️⃣ TÍNH HISTOGRAM:
            → Đếm số pixel cho mỗi mức xám [0-255]
            → Normalize thành phân phối xác suất
        
        2️⃣ THỬ TẤT CẢ NGƯỠNG (t = 1→255):
            Với mỗi ngưỡng t:
            → Chia ảnh thành 2 class:
                • Class 0 (background): [0, t)
                • Class 1 (foreground): [t, 255]
        
        3️⃣ TÍNH BETWEEN-CLASS VARIANCE:
            σ²ʙ(t) = w₀(t) × w₁(t) × [μ₀(t) - μ₁(t)]²
            
            Trong đó:
            • w₀, w₁: Xác suất của class 0, 1
            • μ₀, μ₁: Mean của class 0, 1
        
        4️⃣ CHỌN NGƯỠNG TỐI ƯU:
            t* = argmax σ²ʙ(t)
            → Ngưỡng làm 2 class tách biệt nhất
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào
    
    🔢 LÝ THUYẾT:
        - Between-class variance: Đo độ tách biệt giữa 2 class
        - Variance cao = 2 class phân biệt rõ
        - Tương đương minimize within-class variance
    
    ✅ ƯU ĐIỂM:
        - HOÀN TOÀN TỰ ĐỘNG (không cần tham số)
        - Tối ưu toán học
        - Nhanh (O(256 × n))
        - Reproducible
    
    ❌ NHƯỢC ĐIỂM:
        - Chỉ phù hợp với bimodal histogram
        - Thất bại nếu object/background không cân bằng
        - Nhạy cảm với nhiễu
    
    💡 KHI NÀO DÙNG OTSU:
        ✅ Histogram có 2 đỉnh rõ ràng
        ✅ Object và background có kích thước tương đương
        ✅ Cần phân đoạn tự động
        ❌ Histogram phức tạp (multimodal)
        ❌ Ánh sáng không đều (dùng Adaptive thay thế)
    
    🎯 ỨNG DỤNG:
        - Document scanning (tách chữ khỏi nền)
        - Medical imaging (phân đoạn tế bào)
        - Quality inspection (phát hiện lỗi)
        - Foreground/background separation
    
    📖 THAM KHẢO:
        - Otsu, N. (1979): "A Threshold Selection Method from Gray-Level Histograms"
        - IEEE Trans. Systems, Man, and Cybernetics, Vol. 9, No. 1
        - Citation: 35,000+ (thuật toán kinh điển!)
    
    🔬 SO SÁNH:
        vs Global Threshold: Tự động, không cần đoán ngưỡng
        vs Adaptive: Nhanh hơn, nhưng kém linh hoạt
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
    ⚫ EROSION - Xói mòn vật thể (Morphological Operation)
    
    🎯 MỤC ĐÍCH:
        "Xói" vật thể trắng, làm mảnh đi:
        - Loại bỏ nhiễu nhỏ bên ngoài object
        - Tách các object dính nhau
        - Làm mỏng biên
    
    🔬 THUẬT TOÁN:
        1. Trượt kernel qua từng pixel
        2. Lấy giá trị MINIMUM trong cửa sổ
        3. Gán cho pixel trung tâm
        
        → Kết quả: Pixel trắng chỉ giữ lại nếu TẤT CẢ láng giềng đều trắng
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh binary (đen/trắng)
        kernel_size (int): Kích thước kernel (3, 5, 7)
    
    ✅ HIỆU QUẢ:
        - Loại bỏ nhiễu trắng nhỏ
        - Ngắt kết nối yếu
        - Làm mảnh boundary
    
    💡 ỨNG DỤNG:
        - Khử nhiễu trong ảnh binary
        - Tách đối tượng dính nhau
        - Skeleton extraction (kết hợp nhiều lần)
    
    🔗 KẾT HỢP:
        Erosion + Dilation = Opening (loại nhiễu ngoài)
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
    ⚪ DILATION - Giãn nở vật thể (Morphological Operation)
    
    🎯 MỤC ĐÍCH:
        "Phình" vật thể trắng, làm dày lên:
        - Lấp các lỗ nhỏ bên trong object
        - Nối các phần gần nhau
        - Làm dày biên
    
    🔬 THUẬT TOÁN:
        1. Trượt kernel qua từng pixel
        2. Lấy giá trị MAXIMUM trong cửa sổ
        3. Gán cho pixel trung tâm
        
        → Kết quả: Pixel trắng nếu CÓ ÍT NHẤT 1 láng giềng trắng
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh binary
        kernel_size (int): Kích thước kernel
    
    ✅ HIỆU QUẢ:
        - Lấp lỗ nhỏ
        - Nối kết nối yếu
        - Làm dày boundary
    
    💡 ỨNG DỤNG:
        - Nối văn bản bị đứt
        - Lấp lỗ trong object
        - Tạo buffer zone
    
    🔗 KẾT HỢP:
        Dilation + Erosion = Closing (lấp lỗ trong)
    
    🔬 NGƯỢC LẠI VỚI:
        Erosion (xói mòn) ↔ Dilation (giãn nở)
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
    🔓 OPENING - Xóa nhiễu bên ngoài (Erosion → Dilation)
    
    🎯 MỤC ĐÍCH:
        Loại bỏ nhiễu nhỏ BÊN NGOÀI object mà KHÔNG thay đổi kích thước:
        - Xóa các điểm trắng nhỏ lẻ
        - Làm mịn biên ngoài
        - Ngắt kết nối mảnh
    
    🔬 QUY TRÌNH:
        1. Erosion: Xói mòn → Nhiễu nhỏ biến mất
        2. Dilation: Giãn nở → Phục hồi kích thước ban đầu
        
        → Nhiễu nhỏ không được phục hồi lại!
    
    💡 ỨNG DỤNG:
        - Khử nhiễu trong OCR
        - Làm sạch ảnh scan
        - Tách object riêng biệt
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
    📊 HISTOGRAM EQUALIZATION - Cân bằng histogram để tăng tương phản
    
    🎯 MỤC ĐÍCH:
        Phân bố lại giá trị pixel để:
        - Tăng độ tương phản tự động
        - Sử dụng đầy đủ dải [0-255]
        - Làm nổi chi tiết trong ảnh tối/sáng
    
    🔬 THUẬT TOÁN:
        
        1️⃣ TÍNH HISTOGRAM:
            h(i) = số pixel có giá trị i
        
        2️⃣ TÍNH CDF (Cumulative Distribution Function):
            CDF(i) = Σ h(j) for j=0 to i
        
        3️⃣ NORMALIZE CDF:
            CDF_norm(i) = 255 × CDF(i) / CDF(255)
        
        4️⃣ MAP PIXEL MỚI:
            output[x,y] = CDF_norm[input[x,y]]
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào
    
    🔢 CÔNG THỨC:
        s = T(r) = (L-1) × Σ p(rⱼ)
        Với:
        - r: giá trị pixel gốc
        - s: giá trị pixel mới
        - L: số mức xám (256)
        - p(r): xác suất của r
    
    ✅ ƯU ĐIỂM:
        - Hoàn toàn tự động
        - Tăng contrast hiệu quả
        - Phù hợp với ảnh tối/sáng quá
    
    ❌ NHƯỢC ĐIỂM:
        - Có thể tăng nhiễu
        - Không phù hợp với ảnh contrast tốt
        - Hiệu ứng "không tự nhiên" với ảnh màu
    
    💡 ỨNG DỤNG:
        - Cải thiện ảnh y khoa (X-ray, MRI)
        - Xử lý ảnh vệ tinh
        - Tăng cường ảnh tối (underexposed)
        - Computer vision preprocessing
    
    🎯 KHI NÀO DÙNG:
        ✅ Ảnh tối hoặc sáng quá
        ✅ Histogram tập trung 1 vùng hẹp
        ✅ Cần tăng contrast tự động
        ❌ Ảnh đã có contrast tốt
        ❌ Cần giữ tone màu tự nhiên
    
    📖 THAM KHẢO:
        - Gonzalez & Woods, Chapter 3.3: Histogram Processing
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
    ✨ SHARPENING - Làm sắc nét ảnh bằng Laplacian kernel
    
    🎯 MỤC ĐÍCH:
        Tăng cường biên và chi tiết:
        - Làm nổi bật đường viền
        - Tăng độ sắc nét
        - Làm rõ texture
    
    🔬 THUẬT TOÁN:
        Sử dụng Laplacian kernel (high-pass filter):
        
        Kernel = [[-1, -1, -1],
                  [-1,  9, -1],
                  [-1, -1, -1]]
        
        Nguyên lý:
        - Phát hiện biến đổi bậc 2 (∇²f)
        - Cộng biên vào ảnh gốc
        - Làm nổi chi tiết
    
    📊 THAM SỐ:
        image (np.ndarray): Ảnh đầu vào
    
    ✅ HIỆU QUẢ:
        - Tăng độ sắc nét nhanh
        - Đơn giản, 1 convolution
        - Làm nổi texture
    
    ⚠️ CHÚ Ý:
        - Có thể tăng nhiễu
        - Không dùng cho ảnh nhiễu nhiều
        - Blur trước khi sharpen nếu cần
    
    💡 ỨNG DỤNG:
        - Cải thiện ảnh blur
        - Tăng cường texture
        - Post-processing cho ảnh chụp
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
