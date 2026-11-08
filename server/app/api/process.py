import time
from typing import Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status
)

from pathlib import Path
from app.api.detect import extract_image_bytes, verify_api_key
from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import ErrorResponse
from app.schemas.tts import ProcessDetectResponse
from app.services.scene_assistant_service import SceneAssistantError, describe_scene
from app.services.tts_service import tts_service
from app.services.yolo_service import yolo_service
from app.utils.image_utils import (
    decode_image,
    draw_annotated_image,
    save_annotated_image,
    validate_image_bytes,
)

router = APIRouter(prefix="/process", tags=["Integrated Processing (Vision + Voice)"])


@router.post(
    "/detect",
    response_model=ProcessDetectResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Tệp ảnh không hợp lệ hoặc dữ liệu bị hỏng"},
        413: {"model": ErrorResponse, "description": "Dung lượng ảnh vượt quá 5MB"},
        415: {"model": ErrorResponse, "description": "Định dạng không được hỗ trợ (chỉ JPEG/PNG)"},
        500: {"model": ErrorResponse, "description": "Lỗi nội bộ khi chạy suy luận AI"}
    },
    summary="Nhận ảnh → YOLO-World → Trợ lý AI mô tả cảnh → Giọng nói",
    description=(
        "Quy trình toàn diện cho ESP32-CAM và ứng dụng thông minh:\n"
        "1. Nhận ảnh JPEG (qua multipart hoặc raw body nhị phân).\n"
        "2. Phát hiện vật thể, vật cản và vị trí (trái/giữa/phải) bằng YOLO-World Small.\n"
        "3. Trợ lý AI nhìn ảnh và đối chiếu với các phát hiện YOLO để mô tả ngắn bằng tiếng Việt.\n"
        "4. Gọi Edge-TTS tạo file âm thanh MP3 tương ứng.\n"
        "5. Trả về kết quả nhận diện kèm URL để ESP32-CAM hoặc trình duyệt tải file MP3."
    )
)
async def process_detect_endpoint(
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
        description="Lưu ảnh có Bounding Box vào debug_images để kiểm thử."
    ),
    _auth: None = Depends(verify_api_key)
):
    total_start_time = time.perf_counter()

    # 1. Trích xuất và kiểm tra kích thước / tính toàn vẹn của ảnh
    image_bytes = await extract_image_bytes(request, file)
    validate_image_bytes(image_bytes, settings.MAX_IMAGE_SIZE_BYTES)

    # 2. Giải mã ảnh sang mảng NumPy RGB
    img_rgb = decode_image(image_bytes)
    h, w, _ = img_rgb.shape

    # 3. Chạy nhận diện bằng mô hình chuẩn production (YOLO-World Small)
    engine_name = Path(settings.YOLO_MODEL_PATH).stem
    conf_thresh = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD
    try:
        analysis, inf_time_ms = await yolo_service.analyze_scene(
            img_rgb=img_rgb,
            conf_thresh=conf_thresh
        )
    except Exception as e:
        logger.error(f"Error during AI detection ({engine_name}) in process/detect: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi trong quá trình suy luận mô hình AI ({engine_name}): {str(e)}"
        )

    objects = analysis.all_objects
    description_error: Optional[str] = None
    mentioned_ids: set[int] = set()
    try:
        scene = await describe_scene(img_rgb, objects, client=request.app.state.gemini_client)
        description = scene.description
        mentioned_ids = set(scene.mentioned_ids)
    except SceneAssistantError as exc:
        description_error = str(exc)
        description = "Trợ lý chưa thể mô tả cảnh phía trước."
        logger.warning("Scene assistant unavailable: %s", description_error)

    for index, obj in enumerate(objects):
        obj.is_spoken = index in mentioned_ids
    spoken_objects = [obj for obj in objects if obj.is_spoken]

    # Tùy chọn lưu ảnh debug có Bounding Box
    should_save = save_debug if save_debug is not None else settings.SAVE_DEBUG_IMAGES
    debug_image_url: Optional[str] = None
    if should_save:
        try:
            annotated_bgr = draw_annotated_image(img_rgb, objects)
            saved_path = save_annotated_image(
                img_bgr=annotated_bgr,
                output_dir=settings.IMAGE_DIR,
                filename_prefix="esp32_process"
            )
            debug_image_url = f"/data/images/{saved_path.name}"
        except Exception as e:
            logger.warning(f"Could not save data image: {e}")

    # Read a brief status message if the scene assistant is unavailable.
    audio_id: Optional[str] = None
    audio_url: Optional[str] = None
    audio_error: Optional[str] = None
    tts_time_ms: Optional[float] = None

    try:
        audio_id, _, _, tts_time_ms = await tts_service.generate_speech(
            description, engine=settings.TTS_ENGINE, voice=settings.TTS_VOICE
        )
        audio_url = settings.get_audio_url(audio_id)
    except Exception as exc:
        audio_error = str(exc)
        logger.warning("TTS synthesis failed: %s", audio_error)

    total_time_ms = (time.perf_counter() - total_start_time) * 1000.0

    logger.info(
        f"Process detect completed [{engine_name}]: {len(objects)} total objects, {len(spoken_objects)} spoken | "
        f"Inference: {inf_time_ms:.1f}ms | TTS: {tts_time_ms or 0:.1f}ms | "
        f"Total: {total_time_ms:.1f}ms | Audio: {audio_url or 'ERROR'}"
    )

    return ProcessDetectResponse(
        success=True,
        total_objects=len(objects),
        objects=objects,
        spoken_objects=spoken_objects,
        relationships=analysis.relationships,
        description=description,
        description_error=description_error,
        audio_id=audio_id,
        audio_url=audio_url,
        audio_format="mp3" if audio_url else None,
        audio_error=audio_error,
        inference_time_ms=inf_time_ms,
        tts_time_ms=tts_time_ms,
        total_time_ms=round(total_time_ms, 2),
        image_width=w,
        image_height=h,
        debug_image_url=debug_image_url,
        model_name=engine_name
    )
