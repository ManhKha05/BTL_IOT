from typing import List, Optional
from pydantic import BaseModel, Field


class OCRTextLine(BaseModel):
    """Một dòng chữ nhận dạng được (theo thứ tự đọc trên → dưới, trái → phải)."""
    text: str = Field(..., description="Nội dung dòng chữ (giữ nguyên như OCR nhận dạng)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy trung bình của dòng (0.0 đến 1.0)")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="Khung bao [x1, y1, x2, y2] theo pixel ảnh gốc")


class OCRResponse(BaseModel):
    """Response payload for POST /api/ocr (chỉ OCR, không tạo giọng nói)."""
    success: bool = Field(True, description="Trạng thái xử lý")
    has_text: bool = Field(..., description="True nếu đọc được ít nhất một dòng chữ đủ tin cậy")
    text: str = Field("", description="Văn bản thô nhận dạng được, mỗi dòng ngăn cách bởi '\\n'")
    description: str = Field(..., description="Câu tiếng Việt hoàn chỉnh dùng để đọc (ví dụ 'Nội dung biển hiệu: Nhà thuốc Pharmacity.')")
    context: str = Field(..., description="Loại nội dung: 'sign' (biển hiệu), 'label' (nhãn), 'text' (chung)")
    lines: List[OCRTextLine] = Field(default_factory=list, description="Danh sách dòng chữ đã sắp xếp theo thứ tự đọc")
    low_confidence_count: int = Field(0, description="Số đoạn chữ bị loại do độ tin cậy thấp")
    inference_time_ms: float = Field(..., description="Thời gian chạy OCR (ms)")
    image_width: Optional[int] = Field(None, description="Chiều rộng ảnh (px)")
    image_height: Optional[int] = Field(None, description="Chiều cao ảnh (px)")
    debug_image_url: Optional[str] = Field(None, description="Đường dẫn ảnh đã vẽ khung chữ nếu được bật")
    engine: str = Field("easyocr", description="Engine OCR được sử dụng")


class ProcessOCRResponse(OCRResponse):
    """Response payload for POST /api/process/ocr (OCR + Edge-TTS cho ESP32-CAM)."""
    audio_id: Optional[str] = Field(None, description="UUID của file MP3 (null nếu TTS lỗi)")
    audio_url: Optional[str] = Field(None, description="URL tải file MP3 trên mạng LAN (null nếu TTS lỗi)")
    audio_format: Optional[str] = Field("mp3", description="Định dạng âm thanh")
    audio_error: Optional[str] = Field(None, description="Mô tả lỗi nếu TTS gặp sự cố (ví dụ mất mạng)")
    tts_time_ms: Optional[float] = Field(None, description="Thời gian tổng hợp giọng nói (ms)")
    total_time_ms: float = Field(..., description="Tổng thời gian xử lý request (ms)")
