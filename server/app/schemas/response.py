from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PositionEnum(str, Enum):
    """Position of the detected object across the horizontal axis of the camera frame."""
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class DetectedObject(BaseModel):
    """Individual object detected by the vision model."""
    label: str = Field(..., description="English class label from COCO/custom model (e.g. 'person')")
    label_vi: str = Field(..., description="Vietnamese translation of the label (e.g. 'người')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    position: PositionEnum = Field(..., description="Horizontal position: 'left', 'center', or 'right'")
    bbox: List[int] = Field(..., min_length=4, max_length=4, description="Bounding box [x1, y1, x2, y2] in pixels")
    relative_size: Optional[float] = Field(None, description="Ratio of bounding box area to total frame area")
    priority_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Điểm ưu tiên điều hướng tính từ kích thước, vị trí đáy và hướng di chuyển (0.0 đến 1.0)"
    )
    is_spoken: bool = Field(
        False,
        description="Được thuật toán Object Prioritization lựa chọn để đưa vào câu mô tả âm thanh hay không"
    )
    spoken_label: Optional[str] = Field(
        None,
        description="Nhãn thực tế được sử dụng trong câu đọc tiếng Việt (ví dụ: 'vật thể' khi độ tin cậy thấp)"
    )


class ObjectRelationship(BaseModel):
    """Mối quan hệ không gian và ngữ nghĩa giữa 2 vật thể trong khung hình."""
    subject_idx: int = Field(..., description="Chỉ số (index) của đối tượng chủ thể trong danh sách objects (ví dụ: người)")
    subject_label: str = Field(..., description="Nhãn tiếng Anh của chủ thể (ví dụ: 'person')")
    relation: str = Field(
        ...,
        description="Loại quan hệ: 'person_on_motorcycle', 'person_on_bicycle', 'person_near_motorcycle', 'person_near_bicycle'"
    )
    target_idx: int = Field(..., description="Chỉ số (index) của đối tượng đích trong danh sách objects (ví dụ: xe máy)")
    target_label: str = Field(..., description="Nhãn tiếng Anh của đối tượng đích (ví dụ: 'motorcycle')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Độ tin cậy của mối quan hệ hình học (0.0 đến 1.0)")
    is_certain: bool = Field(..., description="True nếu đủ cơ sở khẳng định người đang đi/điều khiển xe")
    description_vi: Optional[str] = Field(None, description="Mô tả tóm tắt tiếng Việt của quan hệ (ví dụ: 'người đang đi xe máy')")


class DetectResponse(BaseModel):
    """Response payload for POST /api/detect."""
    success: bool = Field(True, description="Whether detection succeeded")
    total_objects: int = Field(..., description="Number of detected objects meeting threshold")
    objects: List[DetectedObject] = Field(default_factory=list, description="Toàn bộ danh sách vật thể phát hiện được (phục vụ debug)")
    spoken_objects: List[DetectedObject] = Field(default_factory=list, description="Danh sách các vật thể được chọn ưu tiên để mô tả bằng giọng nói")
    relationships: List[ObjectRelationship] = Field(
        default_factory=list,
        description="Danh sách quan hệ giữa các vật thể phân tích được (phục vụ debug/kiểm thử)"
    )
    description: str = Field(..., description="Concise, natural Vietnamese scene description")
    inference_time_ms: float = Field(..., description="Time taken by AI model inference in milliseconds")
    image_width: Optional[int] = Field(None, description="Image width in pixels")
    image_height: Optional[int] = Field(None, description="Image height in pixels")
    debug_image_url: Optional[str] = Field(None, description="URL or relative path to annotated image if saved")
    model_name: Optional[str] = Field(None, description="Tên mô hình AI được sử dụng (ví dụ: 'yolov8s' hoặc 'yolov8s-worldv2')")


class HealthResponse(BaseModel):
    """Response payload for GET /health."""
    status: str = "ok"
    service: str = "assistive-ai-server"
    version: str = "1.0.0"
    model_loaded: bool
    model_name: str
    device: str
    server_ip: str
    port: int


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    success: bool = False
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed diagnostic information")
