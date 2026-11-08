"""
Kiểm thử engine Gemini của chức năng 3 bằng phản hồi giả lập (httpx.MockTransport) - không gọi mạng.

Chạy (trong thư mục server/):  python -m pytest tests -v
"""
import asyncio
import json
import sys
from pathlib import Path

import httpx
import numpy as np
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.services.gemini_ocr_service import RESPONSE_SCHEMA  # noqa: E402
from app.services.ocr_service import OCRFragment, ocr_service  # noqa: E402

# Biển "NHÀ THUỐC / AN BÌNH" (ảnh 1000x750), khung theo thang 0-1000 như Gemini trả về
AN_BINH_LINES = [
    {"t": "NHÀ THUỐC", "b": [290, 363, 380, 642], "c": True, "s": "phu"},
    {"t": "AN BÌNH", "b": [340, 166, 600, 798], "c": True, "s": "chinh"},
    {"t": "Dược Phẩm, Mỹ Phẩm - Dụng Cụ Y Khoa", "b": [560, 200, 650, 760], "c": True, "s": "phu"},
    {"t": "ĐC: Tổ 26, Khu Phố 4C, P. Trảng Dài, TP. Biên Hòa", "b": [690, 500, 735, 900], "c": False, "s": "chitiet"},
]


def gemini_reply(lines):
    body = {"candidates": [{
        "finishReason": "STOP",
        "content": {"parts": [{"text": json.dumps({"l": lines}, ensure_ascii=False)}]},
    }]}
    return httpx.Response(200, json=body)


def run_read(handler, monkeypatch, **kwargs):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("test-key"))
    monkeypatch.setattr(settings, "OCR_ENGINE", "auto")
    img = np.zeros((750, 1000, 3), dtype=np.uint8)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ocr_service.read(img, client=client, **kwargs)

    return asyncio.run(go())


def test_gemini_lines_follow_reading_rules(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return gemini_reply(AN_BINH_LINES)

    result = run_read(handler, monkeypatch)
    assert seen["key"] == "test-key"
    assert seen["body"]["contents"][0]["parts"][0]["inlineData"]["mimeType"] == "image/jpeg"
    assert result.engine == "gemini" and result.engine_error is None
    assert result.context == "sign"
    # Mức lấy theo 's' Gemini xếp: 'NHÀ THUỐC' mức Phụ nên được đọc; địa chỉ mức Chi tiết + chưa rõ -> không đọc
    assert result.description == "Biển hiệu: An Bình. Nhà Thuốc. Dược Phẩm, Mỹ Phẩm - Dụng Cụ Y Khoa."
    assert "Biên Hòa" not in result.description


def test_gemini_error_falls_back_to_easyocr(monkeypatch):
    async def fake_easyocr(_img):
        return [OCRFragment("NHÀ THUỐC LONG CHÂU", 0.9, 100, 100, 600, 180)], 0

    monkeypatch.setattr(ocr_service, "_read_easyocr", fake_easyocr)
    result = run_read(lambda request: httpx.Response(429), monkeypatch)
    assert result.engine == "easyocr"
    assert "hạn mức" in result.engine_error
    assert result.description == "Biển hiệu: Nhà Thuốc Long Châu."


def test_easyocr_only_when_engine_disabled(monkeypatch):
    calls = []

    async def fake_easyocr(_img):
        return [OCRFragment("WC", 0.9, 100, 100, 300, 200)], 0

    def handler(request):
        calls.append(request)
        return gemini_reply([])

    monkeypatch.setattr(ocr_service, "_read_easyocr", fake_easyocr)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", SecretStr("test-key"))
    monkeypatch.setattr(settings, "OCR_ENGINE", "easyocr")
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ocr_service.read(img, client=client)

    result = asyncio.run(go())
    assert not calls and result.engine == "easyocr"


def test_response_schema_is_flat():
    """Gemini trả HTTP 400 với schema có $defs / $ref (schema pydantic lồng nhau)."""
    text = json.dumps(RESPONSE_SCHEMA)
    assert "$ref" not in text and "$defs" not in text


def test_gemini_size_overrides_box_height(monkeypatch):
    """Khung Gemini lệch (dòng phụ cao gần bằng dòng chính) nhưng 's' vẫn quyết định mức."""
    lines = [
        {"t": "PHỞ HÀ NỘI", "b": [100, 100, 250, 900], "c": True, "s": "chinh"},
        {"t": "SĐT 0912 345 678", "b": [260, 300, 400, 700], "c": True, "s": "chitiet"},
    ]
    result = run_read(lambda request: gemini_reply(lines), monkeypatch)
    assert result.description == "Biển hiệu: Phở Hà Nội."
    assert [ln.tier for ln in result.lines] == ["main", "detail"]


def test_slow_gemini_request_is_hedged(monkeypatch):
    """Request đầu bị Gemini trả chậm → sau HEDGE giây gửi request dự phòng, lấy kết quả về trước."""
    import time

    calls = []

    async def handler(request):
        calls.append(time.perf_counter())
        if len(calls) == 1:
            await asyncio.sleep(3)  # lần trả lời chậm bất thường
        return gemini_reply(AN_BINH_LINES)

    monkeypatch.setattr(settings, "OCR_GEMINI_HEDGE_SECONDS", 0.1)
    start = time.perf_counter()
    result = run_read(handler, monkeypatch)
    assert result.engine == "gemini"
    assert len(calls) >= 2
    assert time.perf_counter() - start < 2


def test_non_retryable_error_is_not_resent(monkeypatch):
    calls = []

    async def fake_easyocr(_img):
        return [OCRFragment("WC", 0.9, 100, 100, 300, 200)], 0

    def handler(request):
        calls.append(request)
        return httpx.Response(404)

    monkeypatch.setattr(ocr_service, "_read_easyocr", fake_easyocr)
    monkeypatch.setattr(settings, "OCR_GEMINI_HEDGE_SECONDS", 0.05)
    result = run_read(handler, monkeypatch)
    assert len(calls) == 1
    assert result.engine == "easyocr" and "404" in result.engine_error


def test_transient_503_is_retried(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503) if len(calls) == 1 else gemini_reply(AN_BINH_LINES)

    result = run_read(handler, monkeypatch)
    assert len(calls) == 2 and result.engine == "gemini"
