from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.api.detect import verify_api_key
from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import ErrorResponse
from app.schemas.tts import TTSRequest, TTSResponse
from app.services.tts_service import tts_service

router = APIRouter(prefix="", tags=["Text-to-Speech (Edge-TTS)"])


@router.post(
    "/tts",
    response_model=TTSResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Văn bản rỗng hoặc vượt quá 500 ký tự"},
        502: {"model": ErrorResponse, "description": "Lỗi kết nối tới dịch vụ Edge-TTS"},
        504: {"model": ErrorResponse, "description": "Thời gian xử lý Edge-TTS bị timeout"}
    },
    summary="Chuyển văn bản thành giọng nói tiếng Việt",
    description="Nhận câu tiếng Việt, tổng hợp thành file MP3 chất lượng cao bằng Edge-TTS và trả về URL tải file."
)
async def generate_speech_endpoint(
    payload: TTSRequest,
    _auth: None = Depends(verify_api_key)
):
    audio_id, file_path, file_size, duration_ms = await tts_service.generate_speech(
        text=payload.text
    )

    audio_url = settings.get_audio_url(audio_id)

    return TTSResponse(
        success=True,
        text=payload.text,
        audio_id=audio_id,
        audio_url=audio_url,
        format="mp3",
        file_size_bytes=file_size,
        duration_ms=duration_ms
    )


@router.get(
    "/audio/{audio_id}",
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "Dòng dữ liệu âm thanh file MP3 hoàn chỉnh"
        },
        404: {"model": ErrorResponse, "description": "File âm thanh không tồn tại hoặc đã hết hạn"}
    },
    summary="Lấy file âm thanh MP3",
    description="Trả về trực tiếp luồng byte của file MP3. Hỗ trợ trình duyệt nghe trực tiếp (HTML5 audio) và ESP32-CAM tải về phát qua loa."
)
async def get_audio_file_endpoint(audio_id: str):
    file_path = tts_service.get_audio_file(audio_id)

    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tệp âm thanh không tồn tại hoặc đã hết hạn (TTL 30 phút)."
        )

    file_size = file_path.stat().st_size

    # Return FileResponse with inline disposition so browser can play and download
    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=f"{audio_id}.mp3",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Disposition": f'inline; filename="{audio_id}.mp3"',
            "Cache-Control": "public, max-age=1800"
        }
    )
