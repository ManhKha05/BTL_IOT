from typing import List, Optional
from pydantic import BaseModel, Field
from app.schemas.response import DetectedObject, ObjectRelationship


class TTSRequest(BaseModel):
    """Payload for text-to-speech generation (mặc định giọng nữ miền Bắc)."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Nội dung văn bản tiếng Việt cần đọc (tối đa 500 ký tự)"
    )


class TTSResponse(BaseModel):
    """Response payload for POST /api/tts."""
    success: bool = Field(True, description="Trạng thái tạo âm thanh thành công")
    text: str = Field(..., description="Văn bản đã chuyển đổi")
    audio_id: str = Field(..., description="Mã định danh duy nhất (UUID) của file âm thanh")
    audio_url: str = Field(..., description="Đường dẫn HTTP đầy đủ trên mạng LAN để phát hoặc tải file MP3")
    format: str = Field("mp3", description="Định dạng âm thanh")
    file_size_bytes: int = Field(..., description="Dung lượng file MP3 tính bằng bytes")
    duration_ms: float = Field(..., description="Thời gian xử lý tạo giọng nói tính bằng milliseconds")


class ProcessDetectResponse(BaseModel):
    """Response payload for POST /api/process/detect (YOLOv8 + Edge-TTS integrated)."""
    success: bool = Field(True, description="Trạng thái thực hiện chung của request")
    total_objects: int = Field(..., description="Tổng số lượng vật thể phát hiện được")
    objects: List[DetectedObject] = Field(default_factory=list, description="Toàn bộ danh sách vật thể phát hiện được (phục vụ debug)")
    spoken_objects: List[DetectedObject] = Field(default_factory=list, description="Danh sách các vật thể được chọn ưu tiên để mô tả bằng giọng nói")
    relationships: List[ObjectRelationship] = Field(
        default_factory=list,
        description="Danh sách quan hệ giữa các vật thể phân tích được (phục vụ debug/kiểm thử)"
    )
    description: str = Field(..., description="Câu mô tả tiếng Việt tự nhiên tạo từ spoken_objects")
    
    # Audio fields
    audio_id: Optional[str] = Field(None, description="UUID của file MP3 (null nếu TTS lỗi)")
    audio_url: Optional[str] = Field(None, description="URL tải file MP3 trên mạng LAN (null nếu TTS lỗi)")
    audio_format: Optional[str] = Field("mp3", description="Định dạng âm thanh")
    audio_error: Optional[str] = Field(None, description="Mô tả lỗi nếu quá trình TTS gặp sự cố (ví dụ mất mạng)")

    # Diagnostics & Timing
    inference_time_ms: float = Field(..., description="Thời gian suy luận mô hình YOLOv8n (ms)")
    tts_time_ms: Optional[float] = Field(None, description="Thời gian tổng hợp giọng nói Edge-TTS (ms)")
    total_time_ms: float = Field(..., description="Tổng thời gian xử lý toàn bộ request (ms)")
    image_width: Optional[int] = Field(None, description="Chiều rộng ảnh (px)")
    image_height: Optional[int] = Field(None, description="Chiều cao ảnh (px)")
    debug_image_url: Optional[str] = Field(None, description="Đường dẫn ảnh đã vẽ Bounding Box nếu được bật")
    model_name: Optional[str] = Field(None, description="Tên mô hình AI được sử dụng (ví dụ: 'yolov8s' hoặc 'yolov8s-worldv2')")
