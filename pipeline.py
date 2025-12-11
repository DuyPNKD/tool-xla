"""
Processing Pipeline Manager
Quản lý pipeline xử lý ảnh tự động
"""

import numpy as np
from typing import List, Dict, Any, Callable
import json
from datetime import datetime


class ProcessingStep:
    """Đại diện cho một bước xử lý trong pipeline."""
    
    def __init__(self, name: str, function: Callable, params: Dict[str, Any]):
        self.name = name
        self.function = function
        self.params = params
        self.timestamp = datetime.now().isoformat()
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Áp dụng bước xử lý lên ảnh."""
        return self.function(image, **self.params)
    
    def to_dict(self) -> dict:
        """Chuyển thành dictionary để lưu trữ."""
        return {
            'name': self.name,
            'params': self.params,
            'timestamp': self.timestamp
        }


class ImagePipeline:
    """Quản lý pipeline xử lý ảnh."""
    
    def __init__(self, name: str = "Default Pipeline"):
        self.name = name
        self.steps: List[ProcessingStep] = []
        self.history: List[np.ndarray] = []
        self.created_at = datetime.now().isoformat()
    
    def add_step(self, name: str, function: Callable, params: Dict[str, Any]):
        """Thêm một bước xử lý vào pipeline."""
        step = ProcessingStep(name, function, params)
        self.steps.append(step)
        return self
    
    def remove_step(self, index: int):
        """Xóa một bước xử lý."""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
        return self
    
    def clear(self):
        """Xóa tất cả các bước."""
        self.steps.clear()
        self.history.clear()
        return self
    
    def execute(self, image: np.ndarray, save_history: bool = True) -> np.ndarray:
        """
        Thực thi toàn bộ pipeline trên ảnh.
        
        Args:
            image: Ảnh đầu vào
            save_history: Có lưu lịch sử từng bước không
        
        Returns:
            Ảnh sau khi xử lý
        """
        if save_history:
            self.history = [image.copy()]
        
        result = image.copy()
        
        for step in self.steps:
            result = step.apply(result)
            if save_history:
                self.history.append(result.copy())
        
        return result
    
    def get_step_names(self) -> List[str]:
        """Lấy danh sách tên các bước xử lý."""
        return [step.name for step in self.steps]
    
    def export_config(self) -> dict:
        """Export cấu hình pipeline thành JSON."""
        return {
            'name': self.name,
            'created_at': self.created_at,
            'steps': [step.to_dict() for step in self.steps]
        }
    
    def export_python_code(self) -> str:
        """
        Export pipeline thành Python code để tái sử dụng.
        """
        code_lines = [
            "# Auto-generated image processing pipeline",
            f"# Pipeline: {self.name}",
            f"# Created: {self.created_at}",
            "",
            "import numpy as np",
            "import cv2",
            "from PIL import Image",
            "import custom_algorithms as ca",
            "",
            "def process_image(image_path: str) -> np.ndarray:",
            '    """Xử lý ảnh theo pipeline đã định nghĩa."""',
            "    # Đọc ảnh",
            "    image = np.array(Image.open(image_path).convert('RGB'))",
            "    ",
            "    # Áp dụng các bước xử lý",
        ]
        
        for i, step in enumerate(self.steps, 1):
            code_lines.append(f"    # Bước {i}: {step.name}")
            
            # Generate code based on step name
            if step.name == "Grayscale":
                code_lines.append("    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)")
                code_lines.append("    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)")
            
            elif step.name == "Resize":
                scale = step.params.get('scale', 1.0)
                code_lines.append(f"    h, w = image.shape[:2]")
                code_lines.append(f"    new_w, new_h = int(w * {scale}), int(h * {scale})")
                code_lines.append(f"    image = cv2.resize(image, (new_w, new_h))")
            
            elif step.name == "Rotate":
                angle = step.params.get('angle', 0)
                code_lines.append(f"    h, w = image.shape[:2]")
                code_lines.append(f"    center = (w // 2, h // 2)")
                code_lines.append(f"    M = cv2.getRotationMatrix2D(center, {angle}, 1.0)")
                code_lines.append(f"    image = cv2.warpAffine(image, M, (w, h))")
            
            elif "Flip" in step.name:
                mode = step.params.get('mode', 'ngang')
                flip_code = 1 if 'ngang' in mode else 0
                code_lines.append(f"    image = cv2.flip(image, {flip_code})")
            
            elif step.name == "Gaussian Blur":
                ksize = step.params.get('ksize', 5)
                code_lines.append(f"    image = ca.custom_gaussian_blur(image, {ksize})")
            
            elif step.name == "Median Blur":
                ksize = step.params.get('ksize', 5)
                code_lines.append(f"    image = ca.custom_median_filter(image, {ksize})")
            
            elif step.name == "Sharpen":
                code_lines.append(f"    image = ca.custom_sharpen(image)")
            
            elif step.name == "Histogram Equalization":
                code_lines.append(f"    image = ca.custom_histogram_equalization(image)")
            
            elif "Sobel" in step.name:
                direction = 'x' if 'X' in step.name else ('y' if 'Y' in step.name else 'both')
                code_lines.append(f"    image = ca.custom_sobel_operator(image, '{direction}')")
            
            elif step.name == "Canny Edge Detection":
                th1 = step.params.get('th1', 100)
                th2 = step.params.get('th2', 200)
                code_lines.append(f"    image = ca.custom_canny_edge(image, {th1}, {th2})")
            
            elif step.name == "Global Threshold":
                T = step.params.get('T', 127)
                code_lines.append(f"    image = ca.custom_global_threshold(image, {T})")
            
            elif step.name == "Otsu Threshold":
                code_lines.append(f"    image = ca.custom_otsu_threshold(image)")
            
            elif "Adaptive" in step.name:
                method = 'mean' if 'Mean' in step.name else 'gaussian'
                code_lines.append(f"    image = ca.custom_adaptive_threshold(image, 11, 2, '{method}')")
            
            elif step.name in ["Erosion", "Dilation"]:
                ksize = step.params.get('ksize', 5)
                func_name = f"custom_{step.name.lower()}"
                code_lines.append(f"    image = ca.{func_name}(image, {ksize})")
            
            elif step.name in ["Opening", "Closing"]:
                ksize = step.params.get('ksize', 5)
                func_name = f"custom_{step.name.lower()}"
                code_lines.append(f"    image = ca.{func_name}(image, {ksize})")
            
            elif "Cartoon" in step.name:
                code_lines.append(f"    # Cartoon effect")
                code_lines.append(f"    blurred = cv2.bilateralFilter(image, 9, 75, 75)")
                code_lines.append(f"    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)")
                code_lines.append(f"    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)")
                code_lines.append(f"    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)")
                code_lines.append(f"    image = cv2.bitwise_and(blurred, edges_3ch)")
            
            elif "Pencil" in step.name:
                code_lines.append(f"    # Pencil sketch effect")
                code_lines.append(f"    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)")
                code_lines.append(f"    inverted = 255 - gray")
                code_lines.append(f"    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)")
                code_lines.append(f"    sketch = cv2.divide(gray, 255 - blurred, scale=256)")
                code_lines.append(f"    image = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)")
            
            code_lines.append("")
        
        code_lines.extend([
            "    return image",
            "",
            "",
            "if __name__ == '__main__':",
            '    # Sử dụng',
            '    result = process_image("input.jpg")',
            '    Image.fromarray(result).save("output.png")',
        ])
        
        return "\n".join(code_lines)
    
    def __len__(self):
        """Số lượng bước trong pipeline."""
        return len(self.steps)
    
    def __str__(self):
        """Hiển thị thông tin pipeline."""
        return f"Pipeline '{self.name}' with {len(self.steps)} steps"
