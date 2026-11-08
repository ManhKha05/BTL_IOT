"""
Chức năng 3: Đọc chữ trên biển hiệu và nhãn đồ vật.
Thiết bị: nhấn giữ nút 1,5 giây → ESP32-CAM chụp ảnh → POST /api/process/ocr
→ EasyOCR nhận dạng chữ tiếng Việt → ghép thành câu → Edge-TTS → audio_url.
"""
import asyncio
import time
from typing import Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from app.api.detect import extract_image_bytes, verify_api_key
from app.core.config import settings
from app.core.logging import logger
from app.schemas.ocr import OCRResponse, OCRTextLine, ProcessOCRResponse
from app.schemas.response import ErrorResponse
from app.services.ocr_service import OCRResult, draw_ocr_annotations, ocr_service
from app.services.tts_service import tts_service
from app.utils.image_utils import decode_image, save_annotated_image, validate_image_bytes

router = APIRouter(prefix="", tags=["OCR - Đọc chữ biển hiệu & nhãn (EasyOCR)"])

ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Tệp ảnh không hợp lệ hoặc dữ liệu bị hỏng"},
    413: {"model": ErrorResponse, "description": "Dung lượng ảnh vượt quá 5MB"},
    415: {"model": ErrorResponse, "description": "Định dạng không được hỗ trợ (chỉ JPEG/PNG)"},
    500: {"model": ErrorResponse, "description": "Lỗi nội bộ khi chạy OCR"},
    503: {"model": ErrorResponse, "description": "Mô hình OCR chưa sẵn sàng"},
}

CONTEXT_QUERY = Query(
    "auto",
    pattern="^(auto|sign|label|text)$",
    description=(
        "Loại nội dung: 'auto' (tự chọn), 'sign' → 'Biển hiệu: ...' (lọc theo cỡ chữ), "
        "'label' → 'Nhãn: ...' (tên sản phẩm + HSD/NSX/dung tích), 'text' → 'Văn bản: ...' (đọc hết, không lọc)."
    ),
)
SAVE_DEBUG_QUERY = Query(None, description="Lưu ảnh có khung chữ vào data/images để kiểm thử.")
DETAIL_QUERY = Query(False, description="Đọc cả chữ nhỏ mức Chi tiết (SĐT, địa chỉ, quảng cáo...). Mặc định bỏ qua.")
FILE_PARAM = File(None, description="Ảnh JPEG/PNG (tối đa 5MB) - multipart field 'file' hoặc raw body image/jpeg.")


async def _run_ocr(
    request: Request,
    file: Optional[UploadFile],
    context: str,
    save_debug: Optional[bool],
    detail: bool = False,
) -> Tuple[OCRResult, int, int, Optional[str]]:
    image_bytes = await extract_image_bytes(request, file)
    validate_image_bytes(image_bytes, settings.MAX_IMAGE_SIZE_BYTES)
    img_rgb = decode_image(image_bytes)
    h, w, _ = img_rgb.shape

    try:
        result = await ocr_service.read(
            img_rgb,
            context=context,
            detail=detail,
            client=getattr(request.app.state, "gemini_client", None),
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="OCR xử lý quá thời gian cho phép.")
    except (ImportError, RuntimeError) as e:
        logger.error(f"OCR model unavailable: {e}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Mô hình OCR chưa sẵn sàng: {e}")
    except Exception as e:
        logger.error(f"Error during OCR: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Lỗi trong quá trình OCR: {e}")

    debug_image_url: Optional[str] = None
    should_save = save_debug if save_debug is not None else settings.SAVE_DEBUG_IMAGES
    if should_save:
        try:
            saved = save_annotated_image(
                img_bgr=draw_ocr_annotations(img_rgb, result.lines),
                output_dir=settings.IMAGE_DIR,
                filename_prefix="esp32_ocr",
            )
            debug_image_url = f"/data/images/{saved.name}"
            # Lưu kèm ảnh gốc (cùng tên + '_raw') để chạy lại OCR khi cần dò lỗi
            raw_path = saved.with_name(f"{saved.stem}_raw.jpg")
            raw_path.write_bytes(image_bytes)
        except Exception as e:
            logger.warning(f"Could not save OCR debug image: {e}")

    return result, w, h, debug_image_url


def _base_fields(result: OCRResult, w: int, h: int, debug_image_url: Optional[str]) -> dict:
    return dict(
        success=True,
        has_text=result.has_text,
        text=result.text,
        description=result.description,
        context=result.context,
        detail=result.detail,
        lines=[
            OCRTextLine(
                text=ln.text,
                confidence=round(min(max(ln.confidence, 0.0), 1.0), 4),
                bbox=ln.bbox,
                tier=ln.tier,
                spoken=ln.spoken,
                reason=ln.reason,
            )
            for ln in result.lines
        ],
        low_confidence_count=result.low_confidence_count,
        inference_time_ms=result.inference_time_ms,
        image_width=w,
        image_height=h,
        debug_image_url=debug_image_url,
        engine=result.engine,
        engine_error=result.engine_error,
    )


@router.post(
    "/ocr",
    response_model=OCRResponse,
    responses=ERROR_RESPONSES,
    summary="Nhận dạng chữ tiếng Việt trong ảnh (không tạo giọng nói)",
)
async def ocr_endpoint(
    request: Request,
    file: Optional[UploadFile] = FILE_PARAM,
    context: str = CONTEXT_QUERY,
    save_debug: Optional[bool] = SAVE_DEBUG_QUERY,
    detail: bool = DETAIL_QUERY,
    _auth: None = Depends(verify_api_key),
):
    result, w, h, debug_url = await _run_ocr(request, file, context, save_debug, detail)
    logger.info(f"OCR completed: {len(result.lines)} lines in {result.inference_time_ms:.1f}ms | '{result.description}'")
    return OCRResponse(**_base_fields(result, w, h, debug_url))


@router.post(
    "/process/ocr",
    response_model=ProcessOCRResponse,
    responses=ERROR_RESPONSES,
    tags=["Integrated Processing (Vision + Voice)"],
    summary="Chức năng 3: Ảnh → EasyOCR → câu hoàn chỉnh → Edge-TTS giọng nói",
    description=(
        "Dành cho thao tác nhấn giữ nút 1,5 giây trên thiết bị:\n"
        "1. Nhận ảnh JPEG từ ESP32-CAM (multipart hoặc raw body).\n"
        "2. Gemini nhận dạng chữ tiếng Việt (nếu có GEMINI_API_KEY), lỗi thì dùng EasyOCR.\n"
        "3. Chia chữ 3 mức theo cỡ (Chính ≥ 75%, Phụ 25–75%, Chi tiết < 25% so với chữ lớn nhất).\n"
        "4. Đọc Chính → Phụ, trên → dưới, khối trái trước; ví dụ 'Biển hiệu: Bún bò Huế. Bún đậu mẹt.'\n"
        "5. Edge-TTS (dự phòng gTTS) tạo file MP3 và trả về audio_url."
    ),
)
async def process_ocr_endpoint(
    request: Request,
    file: Optional[UploadFile] = FILE_PARAM,
    context: str = CONTEXT_QUERY,
    save_debug: Optional[bool] = SAVE_DEBUG_QUERY,
    detail: bool = DETAIL_QUERY,
    _auth: None = Depends(verify_api_key),
):
    total_start = time.perf_counter()
    result, w, h, debug_url = await _run_ocr(request, file, context, save_debug, detail)

    audio_id: Optional[str] = None
    audio_url: Optional[str] = None
    audio_error: Optional[str] = None
    tts_time_ms: Optional[float] = None
    try:
        audio_id, _, _, tts_time_ms = await tts_service.generate_speech(
            result.description,
            voice=settings.OCR_TTS_VOICE,
            engine=settings.OCR_TTS_ENGINE,
        )
        audio_url = settings.get_audio_url(audio_id)
    except HTTPException as e:
        audio_error = str(e.detail)
    except Exception as e:
        audio_error = str(e)
    if audio_error:
        logger.warning(f"TTS failed for OCR description '{result.description}': {audio_error}")

    total_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
    logger.info(
        f"Process OCR completed: {len(result.lines)} lines | OCR: {result.inference_time_ms:.1f}ms | "
        f"TTS: {tts_time_ms or 0:.1f}ms | Total: {total_ms:.1f}ms | '{result.description}'"
    )

    return ProcessOCRResponse(
        **_base_fields(result, w, h, debug_url),
        audio_id=audio_id,
        audio_url=audio_url,
        audio_format="mp3" if audio_url else None,
        audio_error=audio_error,
        tts_time_ms=tts_time_ms,
        total_time_ms=total_ms,
    )
