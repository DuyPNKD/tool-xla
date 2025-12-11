import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime

# Import custom modules
import custom_algorithms as ca
from metrics import calculate_all_metrics
from pipeline import ImagePipeline
from batch_processor import BatchProcessor

# ==================== CẤU HÌNH TRANG ====================
st.set_page_config(
    page_title="Công cụ Xử lý Ảnh",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS TÙY CHỈNH ====================
st.markdown("""
<style>
    /* Cải thiện sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Card đẹp hơn */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Info box */
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Tiêu đề đẹp hơn */
    h1 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #34495e;
    }
    
    /* Button đẹp hơn */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==================== KHỞI TẠO SESSION STATE ====================
if 'orig_img' not in st.session_state:
    st.session_state.orig_img = None
if 'processed_img' not in st.session_state:
    st.session_state.processed_img = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = ImagePipeline("My Pipeline")
if 'batch_processor' not in st.session_state:
    st.session_state.batch_processor = BatchProcessor()
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# ==================== CÁC HÀM XỬ LÝ ẢNH ====================

def apply_grayscale(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray_3channel

def apply_resize(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized

def apply_rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def apply_flip(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "Flip ngang":
        return cv2.flip(image, 1)
    elif mode == "Flip dọc":
        return cv2.flip(image, 0)
    return image

def apply_gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    blurred = ca.custom_gaussian_blur(image, ksize)
    return blurred

def apply_median_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    blurred = ca.custom_median_filter(image, ksize)
    return blurred

def apply_sharpen(image: np.ndarray) -> np.ndarray:
    sharpened = ca.custom_sharpen(image)
    return sharpened

def apply_hist_equalization(image: np.ndarray) -> np.ndarray:
    equalized = ca.custom_histogram_equalization(image)
    return equalized

def apply_sobel(image: np.ndarray, mode: str) -> np.ndarray:
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
    edges = ca.custom_canny_edge(image, th1, th2)
    return edges

def apply_threshold(image: np.ndarray, T: int) -> np.ndarray:
    thresh = ca.custom_global_threshold(image, T)
    return thresh

def apply_otsu(image: np.ndarray) -> np.ndarray:
    thresh = ca.custom_otsu_threshold(image)
    return thresh

def apply_adaptive_threshold(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "Adaptive Mean Threshold":
        thresh = ca.custom_adaptive_threshold(image, 11, 2, 'mean')
    elif mode == "Adaptive Gaussian Threshold":
        thresh = ca.custom_adaptive_threshold(image, 11, 2, 'gaussian')
    else:
        return image
    return thresh

def apply_morphology(image: np.ndarray, op: str, ksize: int) -> np.ndarray:
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

def apply_cartoon(image: np.ndarray) -> np.ndarray:
    blurred = cv2.bilateralFilter(image, 9, 75, 75)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(blurred, edges_3channel)
    return cartoon

def apply_pencil_sketch(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    sketch_3channel = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    return sketch_3channel

# ==================== HEADER & NAVIGATION ====================
# Header với logo và tiêu đề
col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.image("https://img.icons8.com/fluency/96/000000/image-editing.png", width=80)
with col2:
    st.title("🎨 Công cụ Xử lý Ảnh")
    st.markdown("*Dễ dùng • Mạnh mẽ • Chuyên nghiệp*")
with col3:
    with st.popover("📖 Hướng dẫn"):
        st.markdown("""
        **Bước 1:** Chọn chế độ
        **Bước 2:** Upload ảnh
        **Bước 3:** Chọn hiệu ứng
        **Bước 4:** Xử lý & Tải xuống
        """)

st.markdown("---")

# Navigation ngang với tabs
mode_descriptions = {
    "🖼️ Xử lý ảnh đơn": "Xử lý một ảnh với nhiều hiệu ứng",
    "📦 Xử lý hàng loạt": "Xử lý nhiều ảnh cùng lúc",
    "⚙️ Tạo chuỗi xử lý": "Tạo quy trình xử lý tự động",
    "📊 So sánh chất lượng": "Đo lường PSNR, SSIM giữa 2 ảnh",
    "📜 Lịch sử": "Xem lại các thao tác đã thực hiện"
}

# Tạo tabs cho navigation
mode = st.radio(
    "**🎯 Chọn chế độ xử lý:**",
    list(mode_descriptions.keys()),
    index=0,
    horizontal=True,
    label_visibility="visible"
)

# Hiển thị mô tả chế độ
st.info(f"💡 {mode_descriptions[mode]}")

st.markdown("---")

# ==================== CHẾ ĐỘ 1: SINGLE IMAGE ====================
if mode == "🖼️ Xử lý ảnh đơn":
    # Upload và control panel ngang
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("📤 Bước 1: Chọn ảnh")
        uploaded_file = st.file_uploader(
            "Kéo thả hoặc nhấn để chọn ảnh",
            type=["jpg", "jpeg", "png"],
            help="Hỗ trợ định dạng: JPG, JPEG, PNG",
            label_visibility="collapsed"
        )
    
    
    with col2:
        st.subheader("🎨 Bước 2: Chọn hiệu ứng")
        # Hiệu ứng phổ biến
        quick_methods = {
            "🌫️ Làm mờ": "Làm mờ Gaussian (Gaussian Blur)",
            "✨ Làm sắc nét": "Làm sắc nét (Sharpen)",
            "🎭 Hoạt hình": "Hiệu ứng hoạt hình (Cartoon)",
            "✏️ Vẽ chì": "Hiệu ứng vẽ chì (Pencil Sketch)",
            "⚫ Ảnh xám": "Ảnh xám (Grayscale)",
            "📊 Cân bằng": "Cân bằng Histogram (Histogram Equalization)"
        }
        
        selected_quick = st.selectbox(
            "Chọn hiệu ứng nhanh:",
            [""] + list(quick_methods.values()),
            index=0,
            label_visibility="collapsed"
        )
        
        selected_method = selected_quick if selected_quick else None
    
    with col3:
        st.subheader("⚙️ Tham số")
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        img_array = np.array(pil_image.convert('RGB'))
        st.session_state.orig_img = img_array
        st.session_state.processed_img = None
    
    st.markdown("---")
    
    # Chi tiết hiệu ứng và tham số
    with st.expander("🎯 Chọn hiệu ứng chi tiết (Nâng cao)", expanded=False):
        function_groups = {
        "Tiền xử lý": [
            "Ảnh xám (Grayscale)",
            "Thay đổi kích thước (Resize)",
            "Xoay ảnh (Rotate)",
            "Lật ngang (Flip Horizontal)",
            "Lật dọc (Flip Vertical)"
        ],
        "Tăng cường ảnh": [
            "Làm mờ Gaussian (Gaussian Blur)",
            "Làm mờ Median (Median Blur)",
            "Làm sắc nét (Sharpen)",
            "Cân bằng Histogram (Histogram Equalization)"
        ],
        "Phát hiện biên": [
            "Sobel hướng X (Sobel X)",
            "Sobel hướng Y (Sobel Y)",
            "Sobel tổng hợp (Sobel Magnitude)",
            "Phát hiện biên Canny (Canny Edge)"
        ],
        "Phân ngưỡng": [
            "Phân ngưỡng toàn cục (Global Threshold)",
            "Phân ngưỡng Otsu (Otsu Threshold)",
            "Phân ngưỡng thích ứng Mean (Adaptive Mean)",
            "Phân ngưỡng thích ứng Gaussian (Adaptive Gaussian)"
        ],
        "Hình thái học (Morphology)": [
            "Xói mòn (Erosion)",
            "Giãn nở (Dilation)",
            "Mở (Opening)",
            "Đóng (Closing)"
        ],
        "Hiệu ứng đặc biệt": [
            "Hiệu ứng hoạt hình (Cartoon)",
            "Hiệu ứng vẽ chì (Pencil Sketch)"
        ]
        }
        
        if not selected_method:
            col1, col2 = st.columns(2)
            with col1:
                selected_group = st.selectbox(
                    "Chọn nhóm hiệu ứng:",
                    list(function_groups.keys()),
                    help="Chọn nhóm để xem các hiệu ứng có sẵn"
                )
            with col2:
                selected_method = st.selectbox(
                    "Chọn hiệu ứng cụ thể:",
                    function_groups[selected_group],
                    help="Chọn hiệu ứng bạn muốn áp dụng cho ảnh"
                )
    
    # ===== Bước 3: HIDDEN - Tham số tự động =====
    # Bước điều chỉnh tham số đã được ẩn, sử dụng giá trị mặc định tối ưu
    params = {}
    if "Resize" in selected_method:
        params['scale'] = 1.0
    elif "Rotate" in selected_method:
        params['angle'] = 90
    elif "Gaussian Blur" in selected_method:
        params['ksize'] = 7  # Default to 7x7
    elif "Median Blur" in selected_method:
        params['ksize'] = 5  # Default to 5x5
    elif "Canny" in selected_method:
        params['th1'] = 100
        params['th2'] = 200
    elif "Global Threshold" in selected_method:
        params['T'] = 127
    elif any(x in selected_method for x in ["Erosion", "Dilation", "Opening", "Closing"]):
        params['ksize'] = 5
    
    
    st.markdown("---")
    
    # Nút xử lý và download ngang
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Hiển thị mô tả hiệu ứng
        if selected_method:
            effect_descriptions = {
                "Gaussian Blur": "💡 Làm mờ ảnh để giảm nhiễu, tạo hiệu ứng mềm mại",
                "Sharpen": "💡 Tăng độ sắc nét, làm nổi bật chi tiết trong ảnh",
                "Cartoon": "💡 Biến ảnh thành phong cách hoạt hình",
                "Pencil Sketch": "💡 Tạo hiệu ứng vẽ chì đen trắng nghệ thuật",
                "Grayscale": "💡 Chuyển ảnh màu thành ảnh đen trắng",
                "Histogram Equalization": "💡 Tăng độ tương phản tự động cho ảnh"
            }
            
            for key, desc in effect_descriptions.items():
                if key in selected_method:
                    st.info(desc)
                    break
    
    # Đặt giá trị mặc định cho params
    params = {}
    if "Resize" in selected_method:
        params['scale'] = 1.0
    elif "Rotate" in selected_method:
        params['angle'] = 90
    elif "Gaussian Blur" in selected_method:
        params['ksize'] = 7
    elif "Median Blur" in selected_method:
        params['ksize'] = 5
    elif "Canny" in selected_method:
        params['th1'] = 100
        params['th2'] = 200
    elif "Global Threshold" in selected_method:
        params['T'] = 127
    elif any(x in selected_method for x in ["Erosion", "Dilation", "Opening", "Closing"]):
        params['ksize'] = 5
    
    with col2:
        process_button = st.button("🚀 BẮT ĐẦU XỬ LÝ", type="primary", use_container_width=True)
    
    with col3:
        if st.session_state.processed_img is not None:
            if len(st.session_state.processed_img.shape) == 2:
                processed_pil = Image.fromarray(st.session_state.processed_img, mode='L')
            else:
                processed_pil = Image.fromarray(st.session_state.processed_img)
            
            img_buffer = io.BytesIO()
            processed_pil.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            st.download_button(
                label="💾 TẢI XUỐNG",
                data=img_buffer,
                file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                use_container_width=True,
                type="secondary"
            )
    
    if process_button:
        if st.session_state.orig_img is not None:
            # Tạo progress bar và thông báo
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                import time
                
                # Bước 1: Chuẩn bị
                status_text.text("⏳ Đang chuẩn bị xử lý...")
                progress_bar.progress(10)
                time.sleep(0.1)
                
                # Mapping phương pháp (lấy tên tiếng Anh từ ngoặc)
                def extract_english_name(vn_name):
                    if '(' in vn_name and ')' in vn_name:
                        return vn_name.split('(')[1].split(')')[0]
                    return vn_name
                
                english_method = extract_english_name(selected_method)
                
                # Bước 2: Đang xử lý
                status_text.text(f"🎨 Đang áp dụng hiệu ứng: {selected_method}...")
                progress_bar.progress(30)
                
                method_map = {
                    "Grayscale": lambda img: apply_grayscale(img),
                    "Resize": lambda img: apply_resize(img, params.get('scale', 1.0)),
                    "Rotate": lambda img: apply_rotate(img, params.get('angle', 0)),
                    "Flip Horizontal": lambda img: apply_flip(img, "Flip ngang"),
                    "Flip Vertical": lambda img: apply_flip(img, "Flip dọc"),
                    "Gaussian Blur": lambda img: apply_gaussian_blur(img, params.get('ksize', 5)),
                    "Median Blur": lambda img: apply_median_blur(img, params.get('ksize', 5)),
                    "Sharpen": lambda img: apply_sharpen(img),
                    "Histogram Equalization": lambda img: apply_hist_equalization(img),
                    "Sobel X": lambda img: apply_sobel(img, "Sobel X"),
                    "Sobel Y": lambda img: apply_sobel(img, "Sobel Y"),
                    "Sobel Magnitude": lambda img: apply_sobel(img, "Sobel Magnitude"),
                    "Canny Edge": lambda img: apply_canny(img, params.get('th1', 100), params.get('th2', 200)),
                    "Global Threshold": lambda img: apply_threshold(img, params.get('T', 127)),
                    "Otsu Threshold": lambda img: apply_otsu(img),
                    "Adaptive Mean": lambda img: apply_adaptive_threshold(img, "Adaptive Mean Threshold"),
                    "Adaptive Gaussian": lambda img: apply_adaptive_threshold(img, "Adaptive Gaussian Threshold"),
                    "Erosion": lambda img: apply_morphology(img, "Erosion", params.get('ksize', 5)),
                    "Dilation": lambda img: apply_morphology(img, "Dilation", params.get('ksize', 5)),
                    "Opening": lambda img: apply_morphology(img, "Opening", params.get('ksize', 5)),
                    "Closing": lambda img: apply_morphology(img, "Closing", params.get('ksize', 5)),
                    "Cartoon": lambda img: apply_cartoon(img),
                    "Pencil Sketch": lambda img: apply_pencil_sketch(img)
                }
                
                # Đo thời gian xử lý
                start_time = time.time()
                
                st.session_state.processed_img = method_map[english_method](st.session_state.orig_img)
                
                processing_time = time.time() - start_time
                
                # Bước 3: Hoàn thành xử lý
                status_text.text("✅ Xử lý ảnh hoàn tất!")
                progress_bar.progress(70)
                time.sleep(0.1)
                
                # Lưu vào lịch sử
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'method': selected_method,
                    'params': params
                }
                st.session_state.processing_history.append(history_entry)
                
                # Bước 4: Tính metrics
                status_text.text("📊 Đang tính toán chỉ số chất lượng...")
                progress_bar.progress(85)
                
                if st.session_state.orig_img.shape == st.session_state.processed_img.shape:
                    st.session_state.metrics = calculate_all_metrics(
                        st.session_state.orig_img,
                        st.session_state.processed_img
                    )
                
                # Bước 5: Hoàn tất
                status_text.text("🎉 Hoàn thành!")
                progress_bar.progress(100)
                time.sleep(0.3)
                
                # Xóa progress bar và hiển thị kết quả
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Xử lý thành công trong {processing_time:.2f} giây! Xem kết quả ở bên dưới ⬇️")
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Lỗi khi xử lý: {str(e)}")
                with st.expander("🔍 Chi tiết lỗi (cho developer)"):
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.error("❌ Vui lòng upload ảnh trước!")
    
    st.markdown("---")
    
    # Hiển thị ảnh với UI đẹp hơn
    if st.session_state.orig_img is not None:
        # Hiển thị thông tin ảnh
        img_info = f"📐 Kích thước: {st.session_state.orig_img.shape[1]} x {st.session_state.orig_img.shape[0]} pixels"
        
        # Status
        if st.session_state.processed_img is not None:
            st.success(f"✅ {img_info} | Đã xử lý xong!")
        else:
            st.info(f"ℹ️ {img_info} | Sẵn sàng xử lý")
        
        # Tab view cho dễ so sánh
        tab1, tab2, tab3 = st.tabs(["🔄 So sánh trước/sau", "📷 Ảnh gốc", "✨ Ảnh đã xử lý"])
        
        with tab1:
            if st.session_state.processed_img is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📷 Trước**")
                    st.image(st.session_state.orig_img, use_container_width=True)
                with col2:
                    st.markdown("**✨ Sau**")
                    st.image(st.session_state.processed_img, use_container_width=True)
                
                # Hiển thị metrics nếu có
                if st.session_state.metrics:
                    st.markdown("---")
                    st.subheader("📊 Chỉ số chất lượng")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("PSNR", f"{st.session_state.metrics['PSNR']:.2f} dB")
                    col2.metric("SSIM", f"{st.session_state.metrics['SSIM']*100:.1f}%")
                    col3.metric("MSE", f"{st.session_state.metrics['MSE']:.2f}")
                    col4.metric("MAE", f"{st.session_state.metrics['MAE']:.2f}")
            else:
                st.info("⬅️ Chọn hiệu ứng và nhấn 'BẮT ĐẦU XỬ LÝ' ở sidebar")
        
        with tab2:
            st.image(st.session_state.orig_img, use_container_width=True, caption="Ảnh gốc")
        
        with tab3:
            if st.session_state.processed_img is not None:
                st.image(st.session_state.processed_img, use_container_width=True, caption="Ảnh đã xử lý")
            else:
                st.info("Chưa có ảnh đã xử lý. Hãy chọn hiệu ứng và xử lý!")
    else:
        # Hướng dẫn khi chưa có ảnh
        st.info("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
                <h2>👋 Chào mừng bạn!</h2>
                <p style='font-size: 18px;'>Bắt đầu bằng cách upload ảnh ở sidebar bên trái</p>
                <p style='font-size: 16px;'>📤 Kéo thả hoặc nhấn để chọn ảnh</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Có thể làm gì với công cụ này?")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🎨 Tăng cường ảnh**
            - Làm mờ
            - Làm sắc nét
            - Cân bằng màu sắc
            """)
        with col2:
            st.markdown("""
            **🔍 Phát hiện & Phân tích**
            - Phát hiện biên
            - Phân ngưỡng
            - Xử lý hình thái
            """)
        with col3:
            st.markdown("""
            **✨ Hiệu ứng nghệ thuật**
            - Hoạt hình
            - Vẽ chì
            - Nhiều hiệu ứng khác
            """)

# ==================== CHẾ ĐỘ 2: BATCH PROCESSING ====================
elif mode == "📦 Xử lý hàng loạt":
    st.title("📦 Xử lý Hàng Loạt Ảnh")
    st.markdown("*Upload nhiều ảnh và xử lý cùng lúc - Tiết kiệm thời gian!* ⚡")
    st.markdown("---")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload nhiều ảnh",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.batch_processor.clear()
        for file in uploaded_files:
            pil_image = Image.open(file)
            img_array = np.array(pil_image.convert('RGB'))
            st.session_state.batch_processor.add_image(img_array, file.name)
        
        st.success(f"✅ Đã load {len(uploaded_files)} ảnh")
    
    st.sidebar.markdown("---")
    
    # Chọn phương pháp xử lý batch
    batch_method = st.sidebar.selectbox(
        "Chọn phương pháp",
        [
            "Làm mờ Gaussian (Gaussian Blur)",
            "Làm mờ Median (Median Blur)",
            "Làm sắc nét (Sharpen)",
            "Ảnh xám (Grayscale)",
            "Cân bằng Histogram (Histogram Equalization)"
        ]
    )
    
    batch_params = {}
    if "Gaussian Blur" in batch_method or "Median Blur" in batch_method:
        batch_params['ksize'] = st.sidebar.selectbox("Kích thước kernel", [3, 5, 7, 9], index=2)
    
    if st.sidebar.button("🚀 Xử lý tất cả", type="primary", use_container_width=True):
        if len(st.session_state.batch_processor) > 0:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            start_time = time.time()
            
            status_text.text("⏳ Đang chuẩn bị xử lý hàng loạt...")
            progress_bar.progress(10)
            time.sleep(0.3)
            
            total_imgs = len(st.session_state.batch_processor)
            status_text.text(f"🎨 Đang xử lý {total_imgs} ảnh...")
            progress_bar.progress(30)
            
            # Lấy tên tiếng Anh từ trong ngoặc
            def extract_english(name):
                if '(' in name:
                    return name.split('(')[1].split(')')[0]
                return name
            
            english_name = extract_english(batch_method)
            
            method_map = {
                "Gaussian Blur": lambda img, **p: apply_gaussian_blur(img, p.get('ksize', 5)),
                "Median Blur": lambda img, **p: apply_median_blur(img, p.get('ksize', 5)),
                "Sharpen": lambda img, **p: apply_sharpen(img),
                "Grayscale": lambda img, **p: apply_grayscale(img),
                "Histogram Equalization": lambda img, **p: apply_hist_equalization(img)
            }
            
            st.session_state.batch_processor.process_all(
                method_map[english_name],
                **batch_params
            )
            
            status_text.text("✅ Hoàn tất xử lý tất cả ảnh")
            progress_bar.progress(100)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Cleanup
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Đã xử lý {total_imgs} ảnh trong {processing_time:.2f} giây!")
        else:
            st.sidebar.error("❌ Chưa có ảnh nào!")
    
    # Download ZIP
    if len(st.session_state.batch_processor.processed_images) > 0:
        zip_data = st.session_state.batch_processor.create_zip()
        if zip_data:
            st.sidebar.download_button(
                label="📥 Tải xuống ZIP",
                data=zip_data,
                file_name="processed_batch.zip",
                mime="application/zip",
                use_container_width=True
            )
        
        # Thống kê
        stats = st.session_state.batch_processor.get_statistics()
        st.sidebar.markdown("### 📊 Thống kê")
        st.sidebar.metric("Tổng ảnh", stats['total_images'])
        st.sidebar.metric("Thời gian trung bình", f"{stats['average_time']:.3f}s")
        st.sidebar.metric("Tổng thời gian", f"{stats['total_time']:.3f}s")
    
    # Hiển thị kết quả
    if len(st.session_state.batch_processor.images) > 0:
        st.subheader(f"📸 Danh sách {len(st.session_state.batch_processor.images)} ảnh")
        
        cols_per_row = 3
        rows = (len(st.session_state.batch_processor.images) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                if img_idx < len(st.session_state.batch_processor.images):
                    with cols[col_idx]:
                        st.image(st.session_state.batch_processor.images[img_idx], 
                                caption=f"Ảnh {img_idx + 1}", 
                                use_container_width=True)
                        
                        # Hiển thị ảnh đã xử lý nếu có
                        if img_idx < len(st.session_state.batch_processor.processed_images):
                            st.image(st.session_state.batch_processor.processed_images[img_idx],
                                    caption="Đã xử lý",
                                    use_container_width=True)

# ==================== CHẾ ĐỘ 3: PIPELINE BUILDER ====================
elif mode == "⚙️ Tạo chuỗi xử lý":
    st.title("⚙️ Tạo Chuỗi Xử Lý Tự Động")
    st.markdown("*Kết hợp nhiều hiệu ứng thành một quy trình - Xử lý chuyên nghiệp!* 🎯")
    st.markdown("---")
    
    # Upload ảnh
    uploaded_file = st.sidebar.file_uploader("Upload ảnh test", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        pil_image = Image.open(uploaded_file)
        img_array = np.array(pil_image.convert('RGB'))
        st.session_state.orig_img = img_array
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ Thêm bước xử lý")
    
    # Chọn phương pháp để thêm
    method_to_add = st.sidebar.selectbox(
        "Chọn phương pháp",
        [
            "Ảnh xám (Grayscale)",
            "Làm mờ Gaussian (Gaussian Blur)",
            "Làm mờ Median (Median Blur)",
            "Làm sắc nét (Sharpen)",
            "Cân bằng Histogram (Histogram Equalization)",
            "Sobel X (Sobel X)",
            "Sobel Y (Sobel Y)",
            "Phát hiện biên Canny (Canny Edge)",
            "Phân ngưỡng toàn cục (Global Threshold)",
            "Phân ngưỡng Otsu (Otsu Threshold)"
        ]
    )
    
    step_params = {}
    if "Gaussian Blur" in method_to_add or "Median Blur" in method_to_add:
        step_params['ksize'] = st.sidebar.selectbox("Kích thước kernel", [3, 5, 7, 9], index=2, key="pipeline_ksize")
    elif "Canny" in method_to_add:
        step_params['th1'] = st.sidebar.slider("Ngưỡng 1 (Threshold 1)", 0, 255, 100, key="pipeline_th1")
        step_params['th2'] = st.sidebar.slider("Ngưỡng 2 (Threshold 2)", 0, 255, 200, key="pipeline_th2")
    elif "Global Threshold" in method_to_add:
        step_params['T'] = st.sidebar.slider("Ngưỡng (Threshold)", 0, 255, 127, key="pipeline_T")
    
    if st.sidebar.button("➕ Thêm vào pipeline"):
        # Lấy tên tiếng Anh
        def get_english(name):
            if '(' in name:
                return name.split('(')[1].split(')')[0]
            return name
        
        english_method_name = get_english(method_to_add)
        
        # Mapping hàm
        func_map = {
            "Grayscale": apply_grayscale,
            "Gaussian Blur": apply_gaussian_blur,
            "Median Blur": apply_median_blur,
            "Sharpen": apply_sharpen,
            "Histogram Equalization": apply_hist_equalization,
            "Sobel X": lambda img: apply_sobel(img, "Sobel X"),
            "Sobel Y": lambda img: apply_sobel(img, "Sobel Y"),
            "Canny Edge": apply_canny,
            "Global Threshold": apply_threshold,
            "Otsu Threshold": apply_otsu
        }
        
        st.session_state.pipeline.add_step(
            method_to_add,
            func_map[english_method_name],
            step_params
        )
        st.sidebar.success(f"✅ Đã thêm: {method_to_add}")
    
    st.sidebar.markdown("---")
    
    # Hiển thị pipeline
    st.subheader("📋 Pipeline hiện tại")
    if len(st.session_state.pipeline) > 0:
        for i, step_name in enumerate(st.session_state.pipeline.get_step_names(), 1):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"{i}. {step_name}")
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.pipeline.remove_step(i - 1)
                    st.rerun()
    else:
        st.info("Pipeline trống. Thêm bước xử lý ở sidebar.")
    
    # Nút thực thi
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ Chạy Pipeline", type="primary", use_container_width=True):
            if st.session_state.orig_img is not None and len(st.session_state.pipeline) > 0:
                with st.spinner("Đang xử lý..."):
                    st.session_state.processed_img = st.session_state.pipeline.execute(
                        st.session_state.orig_img
                    )
                    st.success("✅ Hoàn thành!")
            else:
                st.error("❌ Cần có ảnh và ít nhất 1 bước xử lý!")
    
    with col2:
        if st.button("🗑️ Xóa Pipeline", use_container_width=True):
            st.session_state.pipeline.clear()
            st.rerun()
    
    with col3:
        if st.button("💾 Export Code", use_container_width=True):
            if len(st.session_state.pipeline) > 0:
                code = st.session_state.pipeline.export_python_code()
                st.download_button(
                    label="📥 Tải Python Code",
                    data=code,
                    file_name="pipeline_code.py",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # Hiển thị kết quả
    if st.session_state.orig_img is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Ảnh gốc")
            st.image(st.session_state.orig_img, use_container_width=True)
        
        with col2:
            st.subheader("✨ Kết quả Pipeline")
            if st.session_state.processed_img is not None:
                st.image(st.session_state.processed_img, use_container_width=True)
            else:
                st.info("Chưa chạy pipeline")

# ==================== CHẾ ĐỘ 4: COMPARE & METRICS ====================
elif mode == "📊 So sánh chất lượng":
    st.title("📊 So Sánh & Đo Lường Chất Lượng")
    st.markdown("*Đánh giá chất lượng ảnh với các chỉ số chuyên nghiệp: PSNR, SSIM* 📈")
    st.markdown("---")
    
    # Upload 2 ảnh để so sánh
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Ảnh 1 (Original/Reference)")
        file1 = st.file_uploader("Upload ảnh 1", type=["jpg", "jpeg", "png"], key="img1")
        if file1:
            img1 = np.array(Image.open(file1).convert('RGB'))
            st.image(img1, use_container_width=True)
    
    with col2:
        st.subheader("🖼️ Ảnh 2 (Processed/Compare)")
        file2 = st.file_uploader("Upload ảnh 2", type=["jpg", "jpeg", "png"], key="img2")
        if file2:
            img2 = np.array(Image.open(file2).convert('RGB'))
            st.image(img2, use_container_width=True)
    
    # Tính metrics
    if st.button("📊 Tính toán Metrics", type="primary", use_container_width=True):
        if file1 and file2:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            import time
            start_time = time.time()
            
            status_text.text("⏳ Đang chuẩn bị so sánh...")
            progress_bar.progress(10)
            time.sleep(0.2)
            
            status_text.text("📊 Đang tính toán metrics...")
            progress_bar.progress(40)
            
            try:
                metrics = calculate_all_metrics(img1, img2)
                
                status_text.text("✅ Hoàn tất tính toán")
                progress_bar.progress(100)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Cleanup
                time.sleep(0.3)
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Đã tính toán metrics trong {processing_time:.2f} giây!")
                st.markdown("---")
                
                # Hiển thị metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MSE", f"{metrics['MSE']:.2f}", help="Mean Squared Error - Càng nhỏ càng tốt")
                
                with col2:
                    psnr_val = metrics['PSNR']
                    psnr_color = "normal" if psnr_val < 30 else "normal"
                    st.metric("PSNR", f"{psnr_val:.2f} dB", help="Peak Signal-to-Noise Ratio - >30dB là tốt")
                
                with col3:
                    ssim_val = metrics['SSIM']
                    ssim_percent = f"{ssim_val * 100:.1f}%"
                    st.metric("SSIM", ssim_percent, help="Structural Similarity Index - Càng gần 100% càng giống")
                
                with col4:
                    st.metric("MAE", f"{metrics['MAE']:.2f}", help="Mean Absolute Error - Càng nhỏ càng tốt")
                
                st.markdown("---")
                
                # Giải thích
                st.subheader("📝 Giải thích Metrics")
                st.markdown(f"""
                - **MSE (Mean Squared Error)**: {metrics['MSE']:.2f}
                  - Sai số trung bình bình phương giữa 2 ảnh
                  - Giá trị nhỏ → Ảnh giống nhau
                
                - **PSNR (Peak Signal-to-Noise Ratio)**: {metrics['PSNR']:.2f} dB
                  - Đánh giá chất lượng ảnh sau xử lý
                  - > 40 dB: Chất lượng tuyệt vời
                  - 30-40 dB: Chất lượng tốt
                  - < 30 dB: Chất lượng trung bình
                
                - **SSIM (Structural Similarity Index)**: {ssim_percent}
                  - Đánh giá độ tương đồng về cấu trúc
                  - 100%: Hai ảnh giống hệt nhau
                  - > 90%: Rất giống
                  - 70-90%: Giống khá
                  - < 70%: Khác biệt đáng kể
                
                - **MAE (Mean Absolute Error)**: {metrics['MAE']:.2f}
                  - Sai số tuyệt đối trung bình
                  - Giá trị nhỏ → Ảnh giống nhau
                """)
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
        else:
            st.warning("⚠️ Vui lòng upload đủ 2 ảnh!")

# ==================== CHẾ ĐỘ 5: HISTORY ====================
elif mode == "📜 Lịch sử":
    st.title("📜 Lịch Sử Xử Lý")
    st.markdown("*Xem lại tất cả các thao tác đã thực hiện* 🕒")
    st.markdown("---")
    
    if len(st.session_state.processing_history) > 0:
        st.success(f"Có {len(st.session_state.processing_history)} lịch sử xử lý")
        
        # Hiển thị bảng lịch sử
        for i, entry in enumerate(reversed(st.session_state.processing_history), 1):
            with st.expander(f"#{len(st.session_state.processing_history) - i + 1} - {entry['timestamp']} - {entry['method']}"):
                st.write(f"**Phương pháp:** {entry['method']}")
                st.write(f"**Thời gian:** {entry['timestamp']}")
                if entry['params']:
                    st.write(f"**Tham số:** {entry['params']}")
        
        # Nút xóa lịch sử
        if st.button("🗑️ Xóa toàn bộ lịch sử", type="secondary"):
            st.session_state.processing_history.clear()
            st.rerun()
        
        # Export lịch sử
        if st.button("💾 Export lịch sử JSON"):
            history_json = json.dumps(st.session_state.processing_history, indent=2)
            st.download_button(
                label="📥 Tải JSON",
                data=history_json,
                file_name="processing_history.json",
                mime="application/json"
            )
    else:
        st.info("Chưa có lịch sử xử lý nào")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
    <h3>🎨 Công cụ Xử lý Ảnh Chuyên Nghiệp</h3>
    <p style='font-size: 16px;'>✨ Dễ sử dụng • Mạnh mẽ • Miễn phí</p>
    <p>Tự implement thuật toán | Hỗ trợ xử lý hàng loạt | Đo lường chất lượng</p>
</div>
""", unsafe_allow_html=True)

# Thêm phím tắt và tips
with st.expander("💡 Mẹo & Phím tắt"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Phím tắt:**
        - `Ctrl + S`: Lưu kết quả
        - `Ctrl + Z`: Hoàn tác
        - `F5`: Làm mới trang
        """)
    with col2:
        st.markdown("""
        **Mẹo sử dụng:**
        - Dùng Gaussian Blur trước khi phát hiện biên
        - PSNR > 30dB là chất lượng tốt
        - Dùng Pipeline cho quy trình lặp lại
        """)
