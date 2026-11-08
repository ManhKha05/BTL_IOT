"""
Nhận dạng chữ bằng Gemini (Chức năng 3).

Gemini trả từng dòng chữ kèm khung `box_2d` (thang 0-1000), cờ `clear` và mức cỡ chữ `size`.
Khung của Gemini chỉ là ước lượng (lệch giữa các lần gọi), nên mức cỡ chữ lấy theo `size` do Gemini
nhìn ảnh xếp theo đúng quy tắc 3 mức; khung chỉ dùng để sắp thứ tự đọc. Kết quả được đổi thành
OCRFragment giống EasyOCR, nên thứ tự đọc và cách tạo câu trong ocr_service dùng chung:
- clear=True  → độ tin cậy CLEAR_CONFIDENCE (được đọc)
- clear=False → độ tin cậy UNCLEAR_CONFIDENCE (nằm giữa OCR_TIER_MIN_CONFIDENCE và OCR_MIN_CONFIDENCE:
  vẫn dùng để xác định cỡ chữ lớn nhất và báo "chữ chưa rõ", nhưng không đọc chữ sai)
"""
import asyncio
import base64
import io
from pathlib import Path
from typing import List, Literal, Optional
from urllib.parse import quote

import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.core.config import settings

PROMPT = Path(__file__).with_name("gemini_ocr_prompt.md").read_text(encoding="utf-8")
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_IMAGE_SIDE = 1536
CLEAR_CONFIDENCE = 0.95
UNCLEAR_CONFIDENCE = 0.25
ALLOWED_MEDIA_RESOLUTIONS = {"LOW", "MEDIUM", "HIGH"}


class GeminiOCRError(RuntimeError):
    """Gemini không trả được kết quả OCR hợp lệ. retryable=False: gửi lại cũng vô ích (sai khóa, hết hạn mức...)."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


SIZE_TO_TIER = {"chinh": "main", "phu": "sub", "chitiet": "detail"}


class GeminiTextLine(BaseModel):
    """Tên trường ngắn (t / b / c / s) để Gemini trả ít token hơn → phản hồi nhanh hơn."""
    t: str                                          # text
    b: List[int] = Field(min_length=4, max_length=4)  # box_2d [ymin, xmin, ymax, xmax], thang 0-1000
    c: bool                                         # clear
    s: Optional[Literal["chinh", "phu", "chitiet"]] = None  # mức cỡ chữ do Gemini nhìn ảnh xếp


class GeminiOCRResponse(BaseModel):
    l: List[GeminiTextLine] = Field(default_factory=list, max_length=80)  # noqa: E741  (lines)


# Schema phẳng viết tay: Gemini trả HTTP 400 với schema pydantic có $defs / $ref lồng nhau
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "l": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "string"},
                    "b": {"type": "array", "items": {"type": "integer"}},
                    "c": {"type": "boolean"},
                    "s": {"type": "string", "enum": ["chinh", "phu", "chitiet"]},
                },
                "required": ["t", "b", "c", "s"],
            },
        },
    },
    "required": ["l"],
}


def gemini_ocr_enabled() -> bool:
    engine = settings.OCR_ENGINE
    has_key = bool(settings.GEMINI_API_KEY.get_secret_value().strip())
    return engine == "gemini" or (engine == "auto" and has_key)


def _encode_jpeg_base64(img_rgb: np.ndarray) -> str:
    image = Image.fromarray(img_rgb)
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def lines_to_fragments(lines: List[GeminiTextLine], width: int, height: int) -> list:
    """Đổi khung 0-1000 của Gemini sang pixel ảnh gốc; bỏ khung sai hoặc dòng rỗng."""
    from app.services.ocr_service import OCRFragment, _has_meaningful_chars, _normalize_fragment_text

    fragments = []
    for line in lines:
        text = _normalize_fragment_text(line.t)
        if not text or not _has_meaningful_chars(text):
            continue
        ymin, xmin, ymax, xmax = (min(max(v, 0), 1000) for v in line.b)
        if ymax <= ymin or xmax <= xmin:
            continue
        fragments.append(OCRFragment(
            text=text,
            confidence=CLEAR_CONFIDENCE if line.c else UNCLEAR_CONFIDENCE,
            x1=int(xmin / 1000 * width),
            y1=int(ymin / 1000 * height),
            x2=int(xmax / 1000 * width),
            y2=int(ymax / 1000 * height),
            tier_hint=SIZE_TO_TIER.get(line.s) if line.s else None,
        ))
    return fragments


async def _request_once(
    client: httpx.AsyncClient, key: str, encoded: str, media_resolution: str, width: int, height: int
) -> list:
    """Một lần gọi Gemini → danh sách OCRFragment. Lỗi được đổi thành GeminiOCRError."""
    try:
        model = quote(settings.OCR_GEMINI_MODEL.strip(), safe="")
        response = await client.post(
            f"{GEMINI_MODELS_URL}/{model}:generateContent",
            headers={"x-goog-api-key": key},
            timeout=settings.OCR_GEMINI_TIMEOUT_SECONDS,
            json={
                "systemInstruction": {"parts": [{"text": PROMPT}]},
                "contents": [{
                    "role": "user",
                    "parts": [{"inlineData": {"mimeType": "image/jpeg", "data": encoded}}],
                }],
                "generationConfig": {
                    "mediaResolution": f"MEDIA_RESOLUTION_{media_resolution}",
                    "maxOutputTokens": 4096,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": RESPONSE_SCHEMA,
                },
            },
        )
        response.raise_for_status()
        candidate = response.json()["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise ValueError("Incomplete OCR response")
        texts = [
            part["text"]
            for part in candidate["content"]["parts"]
            if "text" in part and not part.get("thought")
        ]
        result = GeminiOCRResponse.model_validate_json("".join(texts))
        return lines_to_fragments(result.l, width, height)
    except httpx.TimeoutException as exc:
        raise GeminiOCRError("Gemini OCR phản hồi quá thời gian cho phép.") from exc
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code == 429:
            raise GeminiOCRError("Gemini đã chạm hạn mức yêu cầu. Hãy thử lại sau.", retryable=False) from exc
        # 400/401/403/404: sai khóa, sai model, sai yêu cầu → gửi lại cũng vô ích
        raise GeminiOCRError(f"Gemini trả lỗi HTTP {code}.", retryable=code >= 500) from exc
    except httpx.RequestError as exc:
        raise GeminiOCRError("Không kết nối được Gemini.") from exc
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise GeminiOCRError("Gemini chưa trả được kết quả OCR hợp lệ.") from exc


async def _hedged(make_request) -> list:
    """
    Gửi request dự phòng song song khi Gemini trả lời chậm.
    Đo thực tế: ~80% lần gọi mất 2-3 giây nhưng thỉnh thoảng vọt lên 18-42 giây ngẫu nhiên.
    Sau mỗi OCR_GEMINI_HEDGE_SECONDS chưa có kết quả (hoặc ngay khi một request lỗi tạm thời),
    gửi thêm một request, tối đa OCR_GEMINI_MAX_PARALLEL; lấy kết quả về sớm nhất, hủy phần còn lại.
    """
    max_parallel = max(1, settings.OCR_GEMINI_MAX_PARALLEL)
    pending: set = set()
    started = 0
    last_error: Optional[GeminiOCRError] = None
    try:
        while True:
            if started < max_parallel:
                pending.add(asyncio.create_task(make_request()))
                started += 1
            elif not pending:
                raise last_error or GeminiOCRError("Gemini chưa trả được kết quả OCR hợp lệ.")
            wait = settings.OCR_GEMINI_HEDGE_SECONDS if started < max_parallel else None
            done, pending = await asyncio.wait(pending, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = task.exception()
                if error is None:
                    return task.result()
                last_error = error if isinstance(error, GeminiOCRError) else GeminiOCRError(str(error))
                if not last_error.retryable:
                    raise last_error
    finally:
        for task in pending:
            task.cancel()


async def gemini_read_fragments(img_rgb: np.ndarray, client: httpx.AsyncClient) -> list:
    key = settings.GEMINI_API_KEY.get_secret_value().strip()
    if not key:
        raise GeminiOCRError("Chưa cấu hình GEMINI_API_KEY trong server/.env.", retryable=False)
    media_resolution = settings.OCR_GEMINI_MEDIA_RESOLUTION.strip().upper()
    if media_resolution not in ALLOWED_MEDIA_RESOLUTIONS:
        raise GeminiOCRError("OCR_GEMINI_MEDIA_RESOLUTION phải là LOW, MEDIUM hoặc HIGH.", retryable=False)

    height, width = img_rgb.shape[:2]
    try:
        encoded = await asyncio.to_thread(_encode_jpeg_base64, img_rgb)
    except OSError as exc:
        raise GeminiOCRError("Không chuẩn bị được ảnh cho Gemini.", retryable=False) from exc

    try:
        async with asyncio.timeout(settings.OCR_GEMINI_TIMEOUT_SECONDS):
            return await _hedged(lambda: _request_once(client, key, encoded, media_resolution, width, height))
    except TimeoutError as exc:
        raise GeminiOCRError("Gemini OCR phản hồi quá thời gian cho phép.") from exc
