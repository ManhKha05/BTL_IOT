"""
API router package for assistive AI server endpoints.
"""
from app.api.detect import router as detect_router
from app.api.tts import router as tts_router
from app.api.process import router as process_router

__all__ = ["detect_router", "tts_router", "process_router"]
