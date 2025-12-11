import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
# Import custom algorithms (tự implement)
import custom_algorithms as ca

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(page_title="Image Processing Tool", layout="wide")

# ==================== KHỞI TẠO SESSION STATE ====================
# Lưu trữ ảnh gốc và ảnh đã xử lý
if 'orig_img' not in st.session_state:
    st.session_state.orig_img = None
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None

# ==================== CÁC HÀM XỬ LÝ ẢNH ====================

# --- Nhóm Tiền xử lý ---

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """Chuyển ảnh sang grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray_3channel


def apply_resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Thay đổi kích thước ảnh theo tỷ lệ."""
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized


def apply_rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """Xoay ảnh theo góc cho trước."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def apply_flip(image: np.ndarray, mode: str) -> np.ndarray:
    """Lật ảnh theo chiều ngang hoặc dọc."""
    if mode == "Flip ngang":
        return cv2.flip(image, 1)  # 1 = flip theo trục dọc (ngang)
    elif mode == "Flip dọc":
        return cv2.flip(image, 0)  # 0 = flip theo trục ngang (dọc)
    return image


# --- Nhóm Tăng cường ảnh ---

def apply_gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """Áp dụng bộ lọc Gaussian Blur với kernel size cho trước.
    CUSTOM IMPLEMENTATION - Tự implement từ đầu."""
    # Đảm bảo ksize là số lẻ
    if ksize % 2 == 0:
        ksize += 1
    blurred = ca.custom_gaussian_blur(image, ksize)
    return blurred


def apply_median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """Áp dụng bộ lọc Median Blur với kernel size cho trước.
    CUSTOM IMPLEMENTATION - Tự implement từ đầu."""
    # Đảm bảo ksize là số lẻ
    if ksize % 2 == 0:
        ksize += 1
    blurred = ca.custom_median_filter(image, ksize)
    return blurred


def apply_sharpen(image: np.ndarray) -> np.ndarray:
    """Làm sắc nét ảnh bằng kernel sharpening.
    CUSTOM IMPLEMENTATION - Tự implement convolution từ đầu."""
    sharpened = ca.custom_sharpen(image)
    return sharpened


def apply_hist_equalization(image: np.ndarray) -> np.ndarray:
    """Cân bằng histogram để tăng cường độ tương phản.
    CUSTOM IMPLEMENTATION - Tự tính CDF và mapping từ đầu."""
    equalized = ca.custom_histogram_equalization(image)
    return equalized


# --- Nhóm Phát hiện biên ---

def apply_sobel(image: np.ndarray, mode: str) -> np.ndarray:
    """Phát hiện biên bằng toán tử Sobel theo hướng X, Y hoặc magnitude.
    CUSTOM IMPLEMENTATION - Tự implement Sobel convolution từ đầu."""
    if mode == "Sobel X":
        result = ca.custom_sobel_operator(image, 'x')
    elif mode == "Sobel Y":
        result = ca.custom_sobel_operator(image, 'y')
    elif mode == "Sobel Magnitude":
        result = ca.custom_sobel_operator(image, 'both')
    else:
        return image
    
    return result


def apply_canny(image: np.ndarray, th1: int, th2: int) -> np.ndarray:
    """Phát hiện biên bằng thuật toán Canny với ngưỡng cho trước.
    CUSTOM IMPLEMENTATION - Tự implement toàn bộ 5 bước Canny từ đầu:
    1. Gaussian blur, 2. Sobel gradient, 3. Non-max suppression,
    4. Double threshold, 5. Edge tracking by hysteresis."""
    edges = ca.custom_canny_edge(image, th1, th2)
    return edges


# --- Nhóm Phân ngưỡng ---

def apply_threshold(image: np.ndarray, T: int) -> np.ndarray:
    """Phân ngưỡng toàn cục với ngưỡng T cho trước.
    CUSTOM IMPLEMENTATION - Tự implement từ đầu."""
    thresh = ca.custom_global_threshold(image, T)
    return thresh


def apply_otsu(image: np.ndarray) -> np.ndarray:
    """Phân ngưỡng tự động bằng phương pháp Otsu.
    CUSTOM IMPLEMENTATION - Tự tính between-class variance để tìm ngưỡng tối ưu."""
    thresh = ca.custom_otsu_threshold(image)
    return thresh


def apply_adaptive_threshold(image: np.ndarray, mode: str) -> np.ndarray:
    """Phân ngưỡng thích ứng với phương pháp Mean hoặc Gaussian.
    CUSTOM IMPLEMENTATION - Tự tính threshold cục bộ từng vùng."""
    if mode == "Adaptive Mean Threshold":
        thresh = ca.custom_adaptive_threshold(image, 11, 2, 'mean')
    elif mode == "Adaptive Gaussian Threshold":
        thresh = ca.custom_adaptive_threshold(image, 11, 2, 'gaussian')
    else:
        return image
    
    return thresh


# --- Nhóm Morphology ---

def apply_morphology(image: np.ndarray, op: str, ksize: int) -> np.ndarray:
    """Áp dụng các phép toán hình thái học: Erosion, Dilation, Opening, Closing.
    CUSTOM IMPLEMENTATION - Tự implement min/max operations từ đầu."""
    if op == "Erosion":
        result = ca.custom_erosion(image, ksize)
    elif op == "Dilation":
        result = ca.custom_dilation(image, ksize)
    elif op == "Opening":
        result = ca.custom_opening(image, ksize)
    elif op == "Closing":
        result = ca.custom_closing(image, ksize)
    else:
        return image
    
    return result


# --- Nhóm Hiệu ứng ---

def apply_cartoon(image: np.ndarray) -> np.ndarray:
    """Tạo hiệu ứng cartoon cho ảnh."""
    # Làm mờ ảnh
    blurred = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Phát hiện biên
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Kết hợp: làm mờ biên để tạo hiệu ứng cartoon
    cartoon = cv2.bitwise_and(blurred, edges_3channel)
    return cartoon


def apply_pencil_sketch(image: np.ndarray) -> np.ndarray:
    """Tạo hiệu ứng vẽ chì (pencil sketch) cho ảnh."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Đảo ngược ảnh grayscale
    inverted = 255 - gray
    
    # Làm mờ ảnh đảo ngược
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    
    # Trộn màu để tạo hiệu ứng sketch
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    
    sketch_3channel = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return sketch_3channel


# ==================== SIDEBAR - TÙY CHỌN XỬ LÝ ====================

st.sidebar.header("⚙️ Tùy chọn xử lý")

# --- Upload ảnh ---
uploaded_file = st.sidebar.file_uploader(
    "Chọn ảnh",
    type=["jpg", "jpeg", "png"]
)

# Xử lý khi người dùng upload ảnh mới
if uploaded_file is not None:
    # Đọc ảnh bằng PIL và chuyển sang numpy array RGB
    pil_image = Image.open(uploaded_file)
    img_array = np.array(pil_image.convert('RGB'))
    # Lưu ảnh gốc vào session state
    st.session_state.orig_img = img_array
    # Reset ảnh đã xử lý khi upload ảnh mới
    st.session_state.processed_img = None

st.sidebar.divider()

# --- Chọn nhóm chức năng ---
function_groups = {
    "Tiền xử lý": ["Grayscale", "Resize", "Rotate", "Flip ngang", "Flip dọc"],
    "Tăng cường ảnh": ["Gaussian Blur", "Median Blur", "Sharpen", "Histogram Equalization"],
    "Phát hiện biên": ["Sobel X", "Sobel Y", "Sobel Magnitude", "Canny Edge Detection"],
    "Phân ngưỡng": ["Global Threshold", "Otsu Threshold", "Adaptive Mean Threshold", "Adaptive Gaussian Threshold"],
    "Morphology": ["Erosion", "Dilation", "Opening", "Closing"],
    "Hiệu ứng": ["Cartoon Effect", "Pencil Sketch"]
}

selected_group = st.sidebar.selectbox(
    "Chọn nhóm chức năng",
    list(function_groups.keys())
)

# --- Chọn phương pháp cụ thể trong nhóm ---
selected_method = st.sidebar.selectbox(
    "Chọn phương pháp",
    function_groups[selected_group]
)

st.sidebar.divider()

# --- Điều khiển tham số (hiển thị tùy theo phương pháp) ---
params = {}

if selected_method == "Resize":
    params['scale'] = st.sidebar.slider("Tỉ lệ phóng/thu", 0.1, 2.0, 1.0, 0.1)

elif selected_method == "Rotate":
    params['angle'] = st.sidebar.slider("Góc xoay (độ)", 0, 360, 0, 1)

elif selected_method == "Gaussian Blur":
    params['ksize'] = st.sidebar.selectbox("Kích thước kernel", [3, 5, 7, 9], index=2)

elif selected_method == "Median Blur":
    params['ksize'] = st.sidebar.selectbox("Kích thước kernel", [3, 5, 7, 9], index=2)

elif selected_method == "Canny Edge Detection":
    params['th1'] = st.sidebar.slider("Threshold 1", 0, 255, 100, 1)
    params['th2'] = st.sidebar.slider("Threshold 2", 0, 255, 200, 1)

elif selected_method == "Global Threshold":
    params['T'] = st.sidebar.slider("Ngưỡng T", 0, 255, 127, 1)

elif selected_method in ["Erosion", "Dilation", "Opening", "Closing"]:
    params['ksize'] = st.sidebar.selectbox("Kích thước kernel", [3, 5, 7], index=1)

# --- Nút xử lý ảnh ---
if st.sidebar.button("🚀 Xử lý ảnh", type="primary", use_container_width=True):
    if st.session_state.orig_img is not None:
        try:
            # Áp dụng phương pháp xử lý tương ứng
            if selected_method == "Grayscale":
                st.session_state.processed_img = apply_grayscale(st.session_state.orig_img)
            
            elif selected_method == "Resize":
                st.session_state.processed_img = apply_resize(st.session_state.orig_img, params['scale'])
            
            elif selected_method == "Rotate":
                st.session_state.processed_img = apply_rotate(st.session_state.orig_img, params['angle'])
            
            elif selected_method in ["Flip ngang", "Flip dọc"]:
                st.session_state.processed_img = apply_flip(st.session_state.orig_img, selected_method)
            
            elif selected_method == "Gaussian Blur":
                st.session_state.processed_img = apply_gaussian_blur(st.session_state.orig_img, params['ksize'])
            
            elif selected_method == "Median Blur":
                st.session_state.processed_img = apply_median_blur(st.session_state.orig_img, params['ksize'])
            
            elif selected_method == "Sharpen":
                st.session_state.processed_img = apply_sharpen(st.session_state.orig_img)
            
            elif selected_method == "Histogram Equalization":
                st.session_state.processed_img = apply_hist_equalization(st.session_state.orig_img)
            
            elif selected_method in ["Sobel X", "Sobel Y", "Sobel Magnitude"]:
                st.session_state.processed_img = apply_sobel(st.session_state.orig_img, selected_method)
            
            elif selected_method == "Canny Edge Detection":
                st.session_state.processed_img = apply_canny(st.session_state.orig_img, params['th1'], params['th2'])
            
            elif selected_method == "Global Threshold":
                st.session_state.processed_img = apply_threshold(st.session_state.orig_img, params['T'])
            
            elif selected_method == "Otsu Threshold":
                st.session_state.processed_img = apply_otsu(st.session_state.orig_img)
            
            elif selected_method in ["Adaptive Mean Threshold", "Adaptive Gaussian Threshold"]:
                st.session_state.processed_img = apply_adaptive_threshold(st.session_state.orig_img, selected_method)
            
            elif selected_method in ["Erosion", "Dilation", "Opening", "Closing"]:
                st.session_state.processed_img = apply_morphology(st.session_state.orig_img, selected_method, params['ksize'])
            
            elif selected_method == "Cartoon Effect":
                st.session_state.processed_img = apply_cartoon(st.session_state.orig_img)
            
            elif selected_method == "Pencil Sketch":
                st.session_state.processed_img = apply_pencil_sketch(st.session_state.orig_img)
            
            st.sidebar.success("✅ Xử lý ảnh thành công!")
        
        except Exception as e:
            st.sidebar.error(f"❌ Lỗi: {str(e)}")
    else:
        st.sidebar.error("❌ Vui lòng upload ảnh trước!")

st.sidebar.divider()

# --- Nút download ảnh đã xử lý ---
if st.session_state.processed_img is not None:
    # Chuyển numpy array sang PIL Image
    # Xử lý trường hợp ảnh 2D (grayscale) hoặc 3D (RGB)
    if len(st.session_state.processed_img.shape) == 2:
        processed_pil = Image.fromarray(st.session_state.processed_img, mode='L')
    else:
        processed_pil = Image.fromarray(st.session_state.processed_img)
    
    # Tạo buffer để lưu ảnh
    img_buffer = io.BytesIO()
    processed_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Nút download
    st.sidebar.download_button(
        label="💾 Lưu ảnh đã xử lý",
        data=img_buffer,
        file_name="processed_image.png",
        mime="image/png",
        use_container_width=True
    )

# ==================== GIAO DIỆN CHÍNH ====================

st.title("🖼️ Công cụ xử lý ảnh")

# Kiểm tra xem đã có ảnh upload chưa
if uploaded_file is None and st.session_state.orig_img is None:
    st.info("👆 Vui lòng upload ảnh ở sidebar bên trái để bắt đầu.")
else:
    # Tạo 2 cột để hiển thị ảnh gốc và ảnh đã xử lý
    col1, col2 = st.columns(2)
    
    # Cột trái: Ảnh gốc
    with col1:
        st.subheader("📷 Ảnh gốc")
        if st.session_state.orig_img is not None:
            st.image(st.session_state.orig_img, channels="RGB", use_container_width=True)
        else:
            st.warning("Chưa có ảnh gốc.")
    
    # Cột phải: Ảnh sau xử lý
    with col2:
        st.subheader("✨ Ảnh sau xử lý")
        if st.session_state.processed_img is not None:
            # Xử lý hiển thị cho cả ảnh 2D (grayscale) và 3D (RGB)
            processed_display = st.session_state.processed_img
            if len(processed_display.shape) == 2:
                # Nếu là ảnh 2D, chuyển sang 3 kênh để hiển thị
                processed_display = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2RGB)
            st.image(processed_display, channels="RGB", use_container_width=True)
        else:
            st.warning("Chưa có ảnh xử lý, hãy bấm nút 'Xử lý ảnh' ở sidebar.")
