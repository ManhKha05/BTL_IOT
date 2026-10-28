import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.detect import router as detect_router
from app.api.process import router as process_router
from app.api.tts import router as tts_router
from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import HealthResponse
from app.services.tts_service import tts_service
from app.services.yolo_service import yolo_service

# Track server start time
START_TIME = time.time()


async def audio_cleanup_worker():
    """Background task running every 10 minutes to clean up expired audio files."""
    try:
        while True:
            await asyncio.sleep(600)  # 10 minutes
            removed = tts_service.cleanup_expired_files(settings.AUDIO_TTL_MINUTES)
            if removed > 0:
                logger.info(f"Periodic cleanup removed {removed} expired audio files.")
    except asyncio.CancelledError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Manager:
    - Preloads YOLO-World Small production model ONCE when FastAPI starts up.
    - Prepares debug and temporary audio directories.
    - Starts background task for TTL audio cleanup.
    - Releases resources cleanly upon server shutdown.
    """
    logger.info("==========================================================")
    logger.info("🚀 Starting Assistive AI Server for Visually Impaired Device")
    logger.info(f"   Version         : {settings.VERSION}")
    logger.info(f"   Server IP       : {settings.LOCAL_IP}:{settings.PORT}")
    logger.info(f"   Production Model: {settings.MODEL_PATH}")
    logger.info(f"   TTS Voice       : {settings.TTS_VOICE}")
    logger.info("==========================================================")

    # Ensure required directories exist (data/images, data/audio, models)
    settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    settings.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up any leftover expired files from previous runs
    tts_service.cleanup_expired_files(settings.AUDIO_TTL_MINUTES)

    # 1. Load unified YOLO-World Small model once at startup
    try:
        yolo_service.load_model()
    except Exception as e:
        logger.critical(f"FATAL: Could not load production YOLO model at startup: {e}")

    # 2. Start audio cleanup background worker
    cleanup_task = asyncio.create_task(audio_cleanup_worker())
    
    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except (asyncio.CancelledError, Exception):
        pass


# Create FastAPI application instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "AI Server for Wearable Device Supporting Visually Impaired Users.\n\n"
        "Features:\n"
        "- **Object Detection**: Fast YOLOv8n inference with spatial division (Left, Center, Right).\n"
        "- **Natural Vietnamese Descriptions**: Converts detected objects into natural Vietnamese sentences.\n"
        "- **Edge-TTS Voice Synthesis**: High-quality Vietnamese speech (`vi-VN-HoaiMyNeural`).\n"
        "- **Integrated Processing**: Single-call pipeline (`/api/process/detect`) for ESP32-CAM.\n"
        "- **Interactive Test Page**: Visit `/test` on your laptop to upload photos, see detections, and listen to generated speech."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount data directory for serving ESP32 photos and audio
app.mount("/data/images", StaticFiles(directory=str(settings.IMAGE_DIR)), name="data_images")
app.mount("/debug_images", StaticFiles(directory=str(settings.IMAGE_DIR)), name="debug_images")

# Mount existing static directory if present
static_dir = settings.BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Lỗi máy chủ nội bộ. Vui lòng kiểm tra log hệ thống.",
            "detail": str(exc) if settings.DEBUG else None
        }
    )


# Include API Routers
app.include_router(detect_router, prefix=settings.API_PREFIX)
app.include_router(tts_router, prefix=settings.API_PREFIX)
app.include_router(process_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint welcoming users and pointing to Swagger documentation and test UI."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "local_ip": settings.LOCAL_IP,
        "port": settings.PORT,
        "docs_url": "/docs",
        "test_ui": "/test",
        "endpoints": {
            "detect": f"{settings.API_PREFIX}/detect",
            "detect_visualize": f"{settings.API_PREFIX}/detect/visualize",
            "tts": f"{settings.API_PREFIX}/tts",
            "audio": f"{settings.API_PREFIX}/audio/{{audio_id}}",
            "process_detect": f"{settings.API_PREFIX}/process/detect",
            "health": "/health"
        }
    }


@app.get("/test", tags=["General"], summary="Trang giao diện kiểm thử trực quan trên laptop")
async def test_page():
    """Serves the interactive web interface to test image detection and listen to Edge-TTS audio."""
    test_html = settings.BASE_DIR / "static" / "test.html"
    if test_html.exists():
        return FileResponse(str(test_html), media_type="text/html")
    return {"message": "Test UI file not found. Please verify static/test.html exists."}


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint to verify server and AI model status."""
    return HealthResponse(
        status="ok" if yolo_service.is_loaded else "degraded",
        service="assistive-ai-server",
        version=settings.VERSION,
        model_loaded=yolo_service.is_loaded,
        model_name=str(settings.MODEL_PATH),
        device=yolo_service.device,
        server_ip=settings.LOCAL_IP,
        port=settings.PORT
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
