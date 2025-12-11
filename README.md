# 🎨 Công cụ Xử lý Ảnh Chuyên Nghiệp

> **Đồ án môn Xử lý Ảnh - Học kỳ 6**  
> Công cụ xử lý ảnh toàn diện với giao diện tiếng Việt, tự implement thuật toán từ đầu

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

---

## 🌟 Điểm nổi bật

### ✨ **100% Tự implement thuật toán**

Không phụ thuộc OpenCV cho các thuật toán xử lý chính - Tất cả đều được code từ đầu với NumPy:

| Thuật toán                    | Chi tiết triển khai                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------ |
| 🟦 **Gaussian Blur**          | Tự tạo kernel 2D theo công thức Gaussian, convolution thủ công                       |
| 🟨 **Median Filter**          | Sliding window với tính median thủ công từng vùng                                    |
| 🔷 **Sobel Edge**             | Tính gradient theo X, Y và magnitude từ kernel Sobel                                 |
| 🔶 **Canny Edge**             | 5 bước đầy đủ: Blur → Gradient → Non-max suppression → Double threshold → Hysteresis |
| 🟧 **Otsu Threshold**         | Tìm ngưỡng tối ưu bằng between-class variance                                        |
| 🟩 **Adaptive Threshold**     | Tính ngưỡng cục bộ cho từng vùng ảnh                                                 |
| ⬛ **Morphology**             | Min/max operations cho Erosion, Dilation, Opening, Closing                           |
| 📊 **Histogram Equalization** | Chuẩn hóa histogram bằng CDF mapping                                                 |
| ⭐ **Sharpening**             | Convolution với Laplacian kernel                                                     |

### 🎯 **Giao diện tiếng Việt thân thiện**

- ✅ Layout ngang hiện đại, dễ sử dụng
- ✅ Không cần điều chỉnh tham số - Tự động tối ưu
- ✅ Thanh tiến trình real-time cho mọi thao tác
- ✅ Hiển thị thời gian xử lý chi tiết

---

## 🚀 5 Chế độ xử lý mạnh mẽ

### 1️⃣ **🖼️ Xử lý ảnh đơn**

```
✨ 20+ hiệu ứng xử lý ảnh
📸 Hiệu ứng nhanh: Làm mờ, Sắc nét, Phát hiện biên, v.v.
📊 Tự động tính metrics (PSNR, SSIM, MSE, MAE)
💾 Tải xuống ảnh ngay lập tức
⏱️ Hiển thị thời gian xử lý real-time
```

### 2️⃣ **📦 Xử lý hàng loạt**

```
📤 Upload nhiều ảnh cùng lúc
🎨 Áp dụng cùng hiệu ứng cho tất cả
📊 Thống kê chi tiết: Tổng thời gian, thời gian TB
📥 Tải xuống tất cả ảnh dạng ZIP
⚡ Thanh tiến trình cho từng ảnh
```

### 3️⃣ **⚙️ Tạo chuỗi xử lý (Pipeline)**

```
🔗 Kết hợp nhiều hiệu ứng thành quy trình
➕ Thêm/xóa bước xử lý linh hoạt
▶️ Chạy toàn bộ pipeline tự động
💻 Export code Python để tái sử dụng
📋 Xem danh sách các bước đã thêm
```

### 4️⃣ **📊 So sánh & đo lường chất lượng**

```
🔍 So sánh 2 ảnh với 4 metrics chuyên nghiệp:
   • MSE (Mean Squared Error) - Sai số bình phương
   • PSNR (Peak Signal-to-Noise Ratio) - Chất lượng tín hiệu
   • SSIM (Structural Similarity) - Độ tương đồng cấu trúc
   • MAE (Mean Absolute Error) - Sai số tuyệt đối
📝 Giải thích chi tiết từng chỉ số
🎯 Đánh giá mức độ giống nhau
```

### 5️⃣ **📜 Lịch sử xử lý**

```
🕒 Lưu tất cả thao tác đã thực hiện
🔎 Xem lại phương pháp và tham số
💾 Export lịch sử dạng JSON
🗑️ Xóa lịch sử khi cần
```

---

## 📦 Cài đặt & Chạy

### **Yêu cầu hệ thống**

- Python 3.8 trở lên
- 4GB RAM (khuyến nghị 8GB)
- Windows / macOS / Linux

### **Bước 1: Clone repository**

```bash
git clone https://github.com/DuyPNKD/tool-xla.git
cd tool-xla
```

### **Bước 2: Cài đặt dependencies**

```bash
pip install -r requirements.txt
```

### **Bước 3: Chạy ứng dụng**

```bash
# 🎯 Khuyến nghị: Phiên bản đầy đủ tính năng
streamlit run app_new.py

# 📌 Phiên bản đơn giản (basic)
streamlit run app.py
```

### **Bước 4: Mở trình duyệt**

Ứng dụng sẽ tự động mở tại: `http://localhost:8501`

---

## 📁 Cấu trúc dự án

```
tool-xla/
│
├── 📱 app_new.py              # ⭐ Ứng dụng chính (Full features)
├── 📱 app.py                  # 📌 Phiên bản basic
│
├── 🧠 custom_algorithms.py    # 🔥 Tự implement 11 thuật toán
├── 📊 metrics.py              # Tính MSE, PSNR, SSIM, MAE
├── ⚙️ pipeline.py             # Quản lý chuỗi xử lý
├── 📦 batch_processor.py      # Xử lý hàng loạt + ZIP export
│
├── 📄 requirements.txt        # Dependencies Python
└── 📖 README.md              # Tài liệu này
```

---

## 🧮 Thuật toán đã triển khai

### 🔷 **1. Filtering & Convolution**

| Thuật toán        | Mô tả                         | Ứng dụng                |
| ----------------- | ----------------------------- | ----------------------- |
| **Gaussian Blur** | Làm mờ với kernel Gaussian 2D | Giảm nhiễu, làm mềm ảnh |
| **Median Filter** | Lọc median trong cửa sổ trượt | Khử nhiễu muối tiêu     |
| **Sharpening**    | Tăng độ sắc nét với Laplacian | Làm nổi bật chi tiết    |

### 🔶 **2. Edge Detection (Phát hiện biên)**

| Thuật toán          | Mô tả                                                      | Ứng dụng                      |
| ------------------- | ---------------------------------------------------------- | ----------------------------- |
| **Sobel X/Y**       | Tính gradient theo hướng X và Y                            | Phát hiện biên theo chiều     |
| **Sobel Magnitude** | Kết hợp Sobel X và Y                                       | Phát hiện biên tổng thể       |
| **Canny Edge**      | 5 bước: Blur → Gradient → Non-max → Threshold → Hysteresis | Phát hiện biên chính xác nhất |

### 🟧 **3. Thresholding (Phân ngưỡng)**

| Thuật toán             | Mô tả                     | Ứng dụng                     |
| ---------------------- | ------------------------- | ---------------------------- |
| **Global Threshold**   | Phân ngưỡng toàn cục      | Tách đối tượng đơn giản      |
| **Otsu's Method**      | Tự động tìm ngưỡng tối ưu | Phân đoạn ảnh tự động        |
| **Adaptive Threshold** | Ngưỡng cục bộ từng vùng   | Xử lý ảnh ánh sáng không đều |

### ⬛ **4. Morphological Operations (Hình thái học)**

| Thuật toán   | Mô tả                 | Ứng dụng            |
| ------------ | --------------------- | ------------------- |
| **Erosion**  | Xói mòn vật thể trắng | Loại bỏ nhiễu nhỏ   |
| **Dilation** | Giãn nở vật thể trắng | Lấp lỗ nhỏ          |
| **Opening**  | Erosion → Dilation    | Xóa nhiễu bên ngoài |
| **Closing**  | Dilation → Erosion    | Lấp lỗ bên trong    |

### 📊 **5. Enhancement (Tăng cường)**

| Thuật toán                 | Mô tả                     | Ứng dụng        |
| -------------------------- | ------------------------- | --------------- |
| **Histogram Equalization** | Chuẩn hóa phân bố độ sáng | Tăng tương phản |

---

## 💻 Export Python Code (Tính năng độc đáo)

Pipeline Builder có thể **tự động sinh code Python** từ các bước xử lý:

### **Ví dụ Pipeline:**

1. Làm mờ Gaussian (7x7)
2. Phát hiện biên Canny (100, 200)
3. Phân ngưỡng Otsu

### **Code được export:**

```python
# Auto-generated by Tool XLA Pipeline Builder
import numpy as np
from PIL import Image
import custom_algorithms as ca

def process_image(image_path: str) -> np.ndarray:
    """
    Pipeline tự động: Gaussian → Canny → Otsu
    """
    # Load ảnh
    image = np.array(Image.open(image_path).convert('RGB'))

    # Bước 1: Làm mờ Gaussian (7x7)
    image = ca.custom_gaussian_blur(image, ksize=7)

    # Bước 2: Phát hiện biên Canny
    image = ca.custom_canny_edge(image, threshold1=100, threshold2=200)

    # Bước 3: Phân ngưỡng Otsu
    image = ca.custom_otsu_threshold(image)

    return image

if __name__ == '__main__':
    result = process_image("input.jpg")
    Image.fromarray(result).save("output.png")
    print("✅ Xử lý xong!")
```

**🎯 Lợi ích:**

- ✅ Tái sử dụng pipeline cho nhiều ảnh
- ✅ Tích hợp vào dự án khác
- ✅ Chạy offline không cần Streamlit
- ✅ Customize code tùy ý

---

## 🛠️ Công nghệ sử dụng

| Công nghệ                                                                     | Phiên bản | Vai trò           |
| ----------------------------------------------------------------------------- | --------- | ----------------- |
| ![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)          | 3.8+      | Ngôn ngữ chính    |
| ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit) | 1.28+     | Framework web UI  |
| ![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue?logo=numpy)            | 1.24+     | Tính toán ma trận |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?logo=opencv)         | 4.8+      | Utility functions |
| ![Pillow](https://img.shields.io/badge/Pillow-10.0+-yellow)                   | 10.0+     | Image I/O         |
| ![SciPy](https://img.shields.io/badge/SciPy-1.11+-orange)                     | 1.11+     | SSIM calculation  |

### 🔑 **Điểm khác biệt:**

- ✅ **NumPy**: Core - Tất cả thuật toán xử lý chính
- ⚠️ **OpenCV**: Chỉ dùng cho flip, rotate, cartoon (không phải thuật toán chính)
- 📊 **SciPy**: Chỉ dùng để tính SSIM metric

---

## 👨‍💻 Tác giả

**DuyPNKD**  
📧 Email: [your-email]  
🔗 GitHub: [@DuyPNKD](https://github.com/DuyPNKD)  
📦 Repository: [tool-xla](https://github.com/DuyPNKD/tool-xla)

---

## 📚 Thông tin đồ án

> **Đồ án môn: Xử lý Ảnh**  
> **Học kỳ:** 6  
> **Khoa:** Công nghệ Thông tin  
> **Năm học:** 2024-2025

### 🎯 **Mục tiêu:**

- ✅ Tự implement các thuật toán xử lý ảnh cơ bản
- ✅ Xây dựng ứng dụng thực tế với giao diện thân thiện
- ✅ Đo lường và đánh giá chất lượng ảnh
- ✅ Tối ưu hiệu suất xử lý

---

## 📄 License

```
MIT License

Copyright (c) 2025 DuyPNKD

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
```

---

## 🌟 Đánh giá & Đóng góp

Nếu thấy project hữu ích, hãy:

- ⭐ **Star** repository
- 🐛 Báo lỗi qua [Issues](https://github.com/DuyPNKD/tool-xla/issues)
- 🔧 Đóng góp code qua [Pull Requests](https://github.com/DuyPNKD/tool-xla/pulls)

---

<div align="center">

**🎨 Made with ❤️ for Image Processing Course**

[![GitHub](https://img.shields.io/badge/GitHub-DuyPNKD-black?logo=github)](https://github.com/DuyPNKD)
[![Streamlit](https://img.shields.io/badge/Built_with-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org/)

</div>
