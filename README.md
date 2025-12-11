# 🎨 Advanced Image Processing Tool

## Tính năng nổi bật

### ✅ Tự implement thuật toán từ đầu

Không chỉ wrapper OpenCV, tất cả các thuật toán chính đều được implement từ đầu:

- ✅ Gaussian Blur (tự tạo kernel và convolution)
- ✅ Median Filter (sliding window median)
- ✅ Sobel Edge Detection (gradient calculation)
- ✅ Canny Edge Detection (đầy đủ 5 bước)
- ✅ Otsu Thresholding (between-class variance)
- ✅ Adaptive Thresholding (local threshold)
- ✅ Morphological Operations (erosion, dilation, opening, closing)
- ✅ Histogram Equalization (CDF mapping)
- ✅ Sharpening (Laplacian kernel)

### 🚀 Tính năng độc đáo

#### 1. 🖼️ Single Image Processing

- Xử lý ảnh đơn với 20+ thuật toán
- Giao diện trực quan, dễ sử dụng
- Preview real-time
- Tải xuống ảnh đã xử lý

#### 2. 📦 Batch Processing

- Xử lý nhiều ảnh cùng lúc
- Hiển thị thống kê thời gian xử lý
- Tải xuống tất cả ảnh dạng ZIP
- Tối ưu cho xử lý hàng loạt

#### 3. ⚙️ Pipeline Builder

- Tạo chuỗi xử lý tự động
- Thêm/xóa các bước xử lý
- **Export Python code** để tái sử dụng
- Lưu và load pipeline

#### 4. 📊 Compare & Metrics

- So sánh 2 ảnh với các chỉ số chuyên nghiệp:
  - **MSE** (Mean Squared Error)
  - **PSNR** (Peak Signal-to-Noise Ratio)
  - **SSIM** (Structural Similarity Index)
  - **MAE** (Mean Absolute Error)
- Giải thích chi tiết từng metrics

#### 5. 📜 History Tracking

- Lưu lịch sử tất cả các thao tác
- Export lịch sử dạng JSON
- Xem lại tham số đã dùng

## Cài đặt

```bash
# Clone repository
git clone https://github.com/DuyPNKD/tool-xla.git
cd tool-xla

# Cài đặt dependencies
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
# Chạy phiên bản mới với tất cả tính năng
streamlit run app_new.py

# Hoặc chạy phiên bản cũ (đơn giản)
streamlit run app.py
```

## Cấu trúc project

```
tool-xla/
├── app.py                  # Phiên bản đơn giản
├── app_new.py             # Phiên bản đầy đủ tính năng
├── custom_algorithms.py   # Tự implement các thuật toán
├── metrics.py             # Tính toán MSE, PSNR, SSIM, MAE
├── pipeline.py            # Quản lý pipeline xử lý
├── batch_processor.py     # Xử lý hàng loạt ảnh
├── requirements.txt       # Dependencies
└── README.md             # File này
```

## Các thuật toán đã implement

### Filtering & Convolution

- Gaussian Blur
- Median Filter
- Sharpening

### Edge Detection

- Sobel Operator (X, Y, Magnitude)
- Canny Edge Detection (5 bước đầy đủ)

### Thresholding

- Global Threshold
- Otsu's Method
- Adaptive Threshold (Mean & Gaussian)

### Morphological Operations

- Erosion
- Dilation
- Opening
- Closing

### Enhancement

- Histogram Equalization

## Demo

### 1. Single Processing

![Single Processing](docs/single.png)

### 2. Batch Processing

![Batch Processing](docs/batch.png)

### 3. Pipeline Builder

![Pipeline Builder](docs/pipeline.png)

### 4. Metrics Comparison

![Metrics](docs/metrics.png)

## Export Python Code

Tool có thể export pipeline thành Python code để tái sử dụng:

```python
# Auto-generated code from pipeline
import numpy as np
import cv2
from PIL import Image
import custom_algorithms as ca

def process_image(image_path: str) -> np.ndarray:
    image = np.array(Image.open(image_path).convert('RGB'))

    # Bước 1: Gaussian Blur
    image = ca.custom_gaussian_blur(image, 5)

    # Bước 2: Canny Edge Detection
    image = ca.custom_canny_edge(image, 100, 200)

    return image

if __name__ == '__main__':
    result = process_image("input.jpg")
    Image.fromarray(result).save("output.png")
```

## Công nghệ sử dụng

- **Streamlit**: Giao diện web
- **NumPy**: Tính toán ma trận
- **OpenCV**: Một số hàm utility (không dùng cho thuật toán chính)
- **PIL/Pillow**: Xử lý ảnh I/O
- **SciPy**: Hỗ trợ tính SSIM

## Tác giả

- **DuyPNKD**
- Repository: [tool-xla](https://github.com/DuyPNKD/tool-xla)

## License

MIT License

## Đồ án

Đây là đồ án môn Xử lý ảnh - Học kỳ 6

- Trường: [Tên trường]
- Khoa: Công nghệ thông tin
- Môn: Xử lý ảnh
