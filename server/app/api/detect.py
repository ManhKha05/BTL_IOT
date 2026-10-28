import io
from pathlib import Path
import time
from typing import Optional
import cv2
from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status
)
from fastapi.responses import Response

from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import DetectResponse, ErrorResponse
from app.services.yolo_service import yolo_service
from app.utils.image_utils import (
    decode_image,
    draw_annotated_image,
    save_annotated_image,
    validate_image_bytes,
)

router = APIRouter(prefix="", tags=["Object Detection (YOLO-World Small)"])


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Optional API Key verification.
    If settings.API_KEY is not configured or empty, access is unrestricted.
    """
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key không hợp lệ hoặc bị thiếu."
        )


async def extract_image_bytes(
    request: Request,
    file: Optional[UploadFile] = None
) -> bytes:
    """
    Extract raw image bytes from either:
    1. multipart/form-data with field 'file' (Standard Swagger / Form upload)
    2. Raw binary request body (Direct HTTP POST from ESP32-CAM)
    """
    if file is not None:
        return await file.read()
    
    # Check if raw binary body was sent
    content_type = request.headers.get("content-type", "")
    if "image/" in content_type or "application/octet-stream" in content_type:
        raw_body = await request.body()
        if raw_body:
            return raw_body
            
    # If no file was provided in either format
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Không tìm thấy tệp ảnh trong request. Vui lòng gửi qua form-data (field 'file') hoặc raw body (Content-Type: image/jpeg)."
    )


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Tệp ảnh không hợp lệ hoặc dữ liệu bị hỏng"},
        413: {"model": ErrorResponse, "description": "Dung lượng ảnh vượt quá 5MB"},
        415: {"model": ErrorResponse, "description": "Định dạng không được hỗ trợ (chỉ JPEG/PNG)"},
        500: {"model": ErrorResponse, "description": "Lỗi nội bộ khi chạy suy luận AI"}
    },
    summary="Nhận diện vật thể bằng YOLO-World Small",
    description="Nhận ảnh từ ESP32-CAM hoặc upload file kiểm thử, phân tích vị trí (trái/giữa/phải) và tạo câu mô tả tiếng Việt."
)
async def detect_objects(
    request: Request,
    file: Optional[UploadFile] = File(
        None,
        description="Tệp ảnh JPEG/PNG (tối đa 5MB). Hỗ trợ upload từ Swagger hoặc ESP32-CAM multipart/raw binary."
    ),
    confidence: Optional[float] = Query(
        None,
        ge=0.1,
        le=0.99,
        description="Ngưỡng tin cậy (Confidence threshold), mặc định 0.35."
    ),
    save_debug: Optional[bool] = Query(
        None,
        description="Lưu ảnh có bounding box vào thư mục data/images để kiểm thử."
    ),
    _auth: None = Depends(verify_api_key)
):
    req_start_time = time.perf_counter()

    # 1. Trích xuất và kiểm tra kích thước / chữ ký ảnh
    image_bytes = await extract_image_bytes(request, file)
    validate_image_bytes(image_bytes, settings.MAX_IMAGE_SIZE_BYTES)

    # 2. Giải mã ảnh sang mảng NumPy RGB
    img_rgb = decode_image(image_bytes)
    h, w, _ = img_rgb.shape

    # 3. Chạy suy luận YOLO-World Small kết hợp thuật toán Object Prioritization
    conf_thresh = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD
    engine_name = "yolov8s-worldv2"
    try:
        objects, spoken_objects, relationships, description, inference_time_ms = await yolo_service.detect(
            img_rgb=img_rgb,
            conf_thresh=conf_thresh
        )
    except Exception as e:
        logger.error(f"Error during AI detection ({engine_name}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình suy luận mô hình AI ({engine_name}): {str(e)}"
        )

    # 4. Tùy chọn lưu ảnh debug có bounding box & phân vùng
    should_save = save_debug if save_debug is not None else settings.SAVE_DEBUG_IMAGES
    debug_image_url: Optional[str] = None
    
    if should_save:
        try:
            annotated_bgr = draw_annotated_image(img_rgb, objects)
            saved_path = save_annotated_image(
                img_bgr=annotated_bgr,
                output_dir=settings.IMAGE_DIR,
                filename_prefix="esp32_detect"
            )
            debug_image_url = f"/data/images/{saved_path.name}"
        except Exception as e:
            logger.warning(f"Failed to save data image: {e}")

    total_req_time_ms = (time.perf_counter() - req_start_time) * 1000.0
    logger.info(
        f"Detect completed [{engine_name}]: {len(objects)} total objects, {len(spoken_objects)} spoken in {inference_time_ms:.1f}ms "
        f"(Total API: {total_req_time_ms:.1f}ms) | Description: '{description}'"
    )

    return DetectResponse(
        success=True,
        total_objects=len(objects),
        objects=objects,
        spoken_objects=spoken_objects,
        relationships=relationships,
        description=description,
        inference_time_ms=inference_time_ms,
        image_width=w,
        image_height=h,
        debug_image_url=debug_image_url,
        model_name=engine_name
    )


@router.post(
    "/detect/visualize",
    summary="Xem trực quan kết quả nhận diện (Trả về ảnh có Bounding Box)",
    description="Nhận ảnh, chạy YOLO-World Small và trả về trực tiếp ảnh JPEG đã vẽ Bounding Box, nhãn tiếng Việt, điểm tin cậy và vạch chia 3 vùng (Trái - Giữa - Phải)."
)
async def detect_visualize(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Tệp ảnh để kiểm tra trực quan"),
    confidence: Optional[float] = Query(None, ge=0.1, le=0.99, description="Ngưỡng tin cậy (mặc định: 0.35)"),
    _auth: None = Depends(verify_api_key)
):
    image_bytes = await extract_image_bytes(request, file)
    validate_image_bytes(image_bytes, settings.MAX_IMAGE_SIZE_BYTES)
    img_rgb = decode_image(image_bytes)

    conf_thresh = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD
    objects, _, _, _, _ = await yolo_service.detect(img_rgb=img_rgb, conf_thresh=conf_thresh)

    annotated_bgr = draw_annotated_image(img_rgb, objects)
    success, encoded_img = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể mã hóa ảnh kết quả."
        )

    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
