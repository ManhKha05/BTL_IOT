import io
import time
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image
from fastapi import HTTPException, status
from app.schemas.response import DetectedObject, PositionEnum


def validate_image_bytes(image_bytes: bytes, max_bytes: int) -> None:
    """
    Validate raw image bytes before decoding:
    - Must not be empty.
    - Must not exceed maximum allowed size.
    - Must have valid JPEG or PNG header signature.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tệp ảnh tải lên trống hoặc không có nội dung."
        )

    if len(image_bytes) > max_bytes:
        size_mb = round(len(image_bytes) / (1024 * 1024), 2)
        max_mb = round(max_bytes / (1024 * 1024), 2)
        raise HTTPException(
            status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
            detail=f"Dung lượng ảnh ({size_mb} MB) vượt quá giới hạn cho phép ({max_mb} MB)."
        )

    # Validate image magic bytes (JPEG or PNG)
    is_jpeg = image_bytes.startswith(b"\xff\xd8\xff")
    is_png = image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if not (is_jpeg or is_png):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Định dạng tệp không được hỗ trợ. Chỉ chấp nhận ảnh JPEG hoặc PNG hợp lệ."
        )


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into a valid RGB numpy array.
    Uses PIL to handle EXIF orientation safely, fallback to OpenCV.
    """
    try:
        # Use PIL to read image and correctly handle potential EXIF orientation
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # Convert to RGB (handles RGBA, Grayscale, etc.)
            rgb_img = pil_img.convert("RGB")
            img_np = np.array(rgb_img)
            
            if img_np.size == 0 or img_np.ndim != 3:
                raise ValueError("Mảng ảnh sau giải mã bị rỗng hoặc sai số chiều.")
            return img_np
    except Exception as e:
        # Fallback to OpenCV decode
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr_img is not None and bgr_img.size > 0:
                return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể giải mã dữ liệu ảnh (dữ liệu có thể bị hỏng): {str(e)}"
        )


def draw_annotated_image(img_rgb: np.ndarray, objects: List[DetectedObject]) -> np.ndarray:
    """
    Draw bounding boxes, labels, confidence scores, and visual vertical zone dividers (Trái - Giữa - Phải).
    Returns BGR image array suitable for saving or streaming.
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = img_bgr.shape

    # Draw vertical divider guide lines for left (33%) and right (67%) zones
    left_line_x = int(w * 0.33)
    right_line_x = int(w * 0.67)

    # Semi-transparent dashed vertical lines for spatial zones
    cv2.line(img_bgr, (left_line_x, 0), (left_line_x, h), (180, 180, 180), 1, cv2.LINE_AA)
    cv2.line(img_bgr, (right_line_x, 0), (right_line_x, h), (180, 180, 180), 1, cv2.LINE_AA)

    # Add zone header labels at top of image
    cv2.putText(img_bgr, "TRAI (LEFT)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img_bgr, "GIUA (CENTER)", (left_line_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img_bgr, "PHAI (RIGHT)", (right_line_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Draw detected objects
    color_map = {
        PositionEnum.LEFT: (255, 128, 0),    # Blue-ish orange
        PositionEnum.CENTER: (0, 220, 0),    # Bright green
        PositionEnum.RIGHT: (0, 165, 255),   # Orange
    }

    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        box_color = color_map.get(obj.position, (0, 255, 0))

        # Distinct thickness for spoken vs filtered objects
        is_spoken = getattr(obj, "is_spoken", False)
        thickness = 3 if is_spoken else 1

        # Draw box rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color, thickness)

        # Label tag: "[DOC/LOC] label_vi (conf%) S:score [pos]"
        status_prefix = "[DOC] " if is_spoken else ""
        score_str = f" S:{obj.priority_score:.2f}" if getattr(obj, "priority_score", None) is not None else ""
        label_text = f"{status_prefix}{obj.label_vi} ({int(obj.confidence * 100)}%){score_str} [{obj.position.value}]"
        
        # Draw background banner for text readability
        font_scale = 0.45
        text_thick = 1
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
        
        banner_y1 = max(0, y1 - text_h - baseline - 4)
        banner_y2 = y1
        cv2.rectangle(img_bgr, (x1, banner_y1), (x1 + text_w + 6, banner_y2), box_color, -1)
        
        # Put text
        cv2.putText(
            img_bgr,
            label_text,
            (x1 + 3, banner_y2 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thick,
            cv2.LINE_AA
        )

        # Draw center point of object
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(img_bgr, (cx, cy), 4, (0, 0, 255), -1)

    return img_bgr


def save_annotated_image(img_bgr: np.ndarray, output_dir: Path, filename_prefix: str = "detect") -> Path:
    """Save annotated BGR image to the target directory and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    file_path = output_dir / f"{filename_prefix}_{timestamp}.jpg"
    cv2.imwrite(str(file_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return file_path
