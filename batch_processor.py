"""
Batch Image Processor
Xử lý hàng loạt nhiều ảnh cùng lúc
"""

import numpy as np
from PIL import Image
import os
from typing import List, Callable, Dict, Any
from datetime import datetime
import zipfile
import io


class BatchProcessor:
    """Xử lý hàng loạt nhiều ảnh."""
    
    def __init__(self):
        self.images: List[np.ndarray] = []
        self.filenames: List[str] = []
        self.processed_images: List[np.ndarray] = []
        self.processing_times: List[float] = []
    
    def add_image(self, image: np.ndarray, filename: str = None):
        """Thêm một ảnh vào batch."""
        self.images.append(image)
        if filename is None:
            filename = f"image_{len(self.images)}.png"
        self.filenames.append(filename)
    
    def clear(self):
        """Xóa tất cả ảnh trong batch."""
        self.images.clear()
        self.filenames.clear()
        self.processed_images.clear()
        self.processing_times.clear()
    
    def process_all(self, processing_func: Callable, **kwargs) -> List[np.ndarray]:
        """
        Xử lý tất cả ảnh trong batch với cùng một hàm xử lý.
        
        Args:
            processing_func: Hàm xử lý ảnh
            **kwargs: Tham số cho hàm xử lý
        
        Returns:
            Danh sách ảnh đã xử lý
        """
        self.processed_images.clear()
        self.processing_times.clear()
        
        for img in self.images:
            start_time = datetime.now()
            processed = processing_func(img, **kwargs)
            end_time = datetime.now()
            
            self.processed_images.append(processed)
            self.processing_times.append((end_time - start_time).total_seconds())
        
        return self.processed_images
    
    def process_with_pipeline(self, pipeline) -> List[np.ndarray]:
        """
        Xử lý tất cả ảnh với một pipeline.
        
        Args:
            pipeline: ImagePipeline object
        
        Returns:
            Danh sách ảnh đã xử lý
        """
        self.processed_images.clear()
        self.processing_times.clear()
        
        for img in self.images:
            start_time = datetime.now()
            processed = pipeline.execute(img, save_history=False)
            end_time = datetime.now()
            
            self.processed_images.append(processed)
            self.processing_times.append((end_time - start_time).total_seconds())
        
        return self.processed_images
    
    def get_statistics(self) -> dict:
        """Lấy thống kê về quá trình xử lý batch."""
        if not self.processing_times:
            return {}
        
        return {
            'total_images': len(self.images),
            'processed_images': len(self.processed_images),
            'total_time': sum(self.processing_times),
            'average_time': np.mean(self.processing_times),
            'min_time': min(self.processing_times),
            'max_time': max(self.processing_times)
        }
    
    def create_zip(self) -> bytes:
        """
        Tạo file ZIP chứa tất cả ảnh đã xử lý.
        
        Returns:
            Nội dung file ZIP dạng bytes
        """
        if not self.processed_images:
            return None
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, (img, filename) in enumerate(zip(self.processed_images, self.filenames)):
                # Chuyển numpy array sang PIL Image
                if len(img.shape) == 2:
                    pil_img = Image.fromarray(img, mode='L')
                else:
                    pil_img = Image.fromarray(img)
                
                # Lưu vào buffer
                img_buffer = io.BytesIO()
                pil_img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                
                # Thêm vào ZIP
                base_name = os.path.splitext(filename)[0]
                zip_file.writestr(f"processed_{base_name}.png", img_buffer.getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def __len__(self):
        """Số lượng ảnh trong batch."""
        return len(self.images)
    
    def __str__(self):
        """Hiển thị thông tin batch."""
        return f"BatchProcessor with {len(self.images)} images"
