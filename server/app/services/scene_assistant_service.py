"""Describe a camera frame using the image and grounded YOLO hints."""

import asyncio
import base64
import io
import json
from pathlib import Path
from urllib.parse import quote

import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.schemas.response import DetectedObject

PROMPT = Path(__file__).with_name("scene_assistant_prompt.md").read_text(encoding="utf-8")
GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
MAX_HINTS = 12
MAX_WORDS = 30
MAX_CHARACTERS = 200


class SceneAssistantError(RuntimeError):
    """A scene description could not be obtained."""


class SceneDescription(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    description: str
    mentioned_ids: list[int] = Field(max_length=3)


def encode_jpeg_base64(img_rgb: np.ndarray) -> str:
    image = Image.fromarray(img_rgb)
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def describe_scene(
    img_rgb: np.ndarray, objects: list[DetectedObject], *, client: httpx.AsyncClient
) -> SceneDescription:
    key = settings.GEMINI_API_KEY.get_secret_value().strip()
    if not key:
        raise SceneAssistantError("Chưa cấu hình GEMINI_API_KEY trong server/.env.")
    media_resolution = settings.SCENE_AI_MEDIA_RESOLUTION.strip().upper()
    if media_resolution not in {"LOW", "MEDIUM"}:
        raise SceneAssistantError("SCENE_AI_MEDIA_RESOLUTION phải là LOW hoặc MEDIUM.")

    height, width = img_rgb.shape[:2]
    hints = [
        {
            "id": index,
            "name_vi": obj.label_vi,
            "zone": obj.position.value,
            "bbox_xyxy_relative": [
                round(coord / size, 4)
                for coord, size in zip(obj.bbox, (width, height, width, height))
            ],
            "confidence": obj.confidence,
        }
        for index, obj in enumerate(objects[:MAX_HINTS])
    ]
    context = json.dumps(
        {"yolo_hints": hints},
        ensure_ascii=False,
    )
    try:
        async with asyncio.timeout(settings.SCENE_AI_TIMEOUT_SECONDS):
            encoded_image = await asyncio.to_thread(encode_jpeg_base64, img_rgb)
            model = quote(settings.SCENE_AI_MODEL.strip(), safe="")
            response = await client.post(
                f"{GEMINI_MODELS_URL}/{model}:generateContent",
                headers={"x-goog-api-key": key},
                timeout=settings.SCENE_AI_TIMEOUT_SECONDS,
                json={
                    "systemInstruction": {"parts": [{"text": PROMPT}]},
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {"text": context},
                            {"inlineData": {"mimeType": "image/jpeg", "data": encoded_image}},
                        ],
                    }],
                    "generationConfig": {
                        "mediaResolution": f"MEDIA_RESOLUTION_{media_resolution}",
                        "maxOutputTokens": 512,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": SceneDescription.model_json_schema(),
                    },
                },
            )
            response.raise_for_status()
        candidate = response.json()["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise ValueError("Incomplete scene description")
        texts = [
            part["text"]
            for part in candidate["content"]["parts"]
            if "text" in part and not part.get("thought")
        ]
        result = SceneDescription.model_validate_json("".join(texts))
        if (
            not result.description
            or len(result.description) > MAX_CHARACTERS
            or len(result.description.split()) > MAX_WORDS
        ):
            raise ValueError("Scene description is empty or too long")
        if any(not 0 <= index < len(hints) for index in result.mentioned_ids):
            raise ValueError("Unknown YOLO hint ID")
        return result
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise SceneAssistantError("Trợ lý AI phản hồi quá thời gian cho phép.") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise SceneAssistantError("Gemini đã chạm hạn mức yêu cầu. Hãy thử lại sau.") from exc
        raise SceneAssistantError(f"Gemini trả lỗi HTTP {exc.response.status_code}.") from exc
    except httpx.RequestError as exc:
        raise SceneAssistantError("Không kết nối được dịch vụ mô tả AI.") from exc
    except OSError as exc:
        raise SceneAssistantError("Không chuẩn bị được ảnh cho trợ lý AI.") from exc
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        raise SceneAssistantError("Trợ lý AI chưa trả được mô tả hợp lệ.") from exc
