import os
import socket
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Determine project root (the directory containing app/, models/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env if present
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()


def get_local_ip() -> str:
    """Retrieve the primary local IPv4 address of the host machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        # Does not actually send data over the network
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class Settings(BaseModel):
    """Application settings with environment variable override support."""
    PROJECT_NAME: str = "Assistive AI Server for Visually Impaired"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Server Networking
    HOST: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"))
    LOCAL_IP: str = Field(default_factory=get_local_ip)

    # AI Vision Model Configuration (YOLO-World Small for production)
    # models/yolov8s-worldv2.pt: 25.9M params, open-vocabulary pedestrian & road obstacle detector
    YOLO_MODEL_PATH: str = Field(
        default_factory=lambda: os.getenv("YOLO_MODEL_PATH", os.getenv("MODEL_PATH", "models/yolov8s-worldv2.pt"))
    )
    MODEL_PATH: str = Field(
        default_factory=lambda: os.getenv("YOLO_MODEL_PATH", os.getenv("MODEL_PATH", "models/yolov8s-worldv2.pt"))
    )
    CONFIDENCE_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.35")))
    
    # Cautious description threshold:
    # When detection confidence is between CONFIDENCE_THRESHOLD and CONFIDENCE_CERTAIN_THRESHOLD,
    # the scene description will use cautious phrasing ("vật thể" / "vật cản") instead of asserting
    # an uncertain specific label (e.g. avoiding falsely claiming a dog is a cow).
    CONFIDENCE_CERTAIN_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("CONFIDENCE_CERTAIN_THRESHOLD", "0.50")))
    CAUTIOUS_DESCRIPTION_ENABLED: bool = Field(default_factory=lambda: os.getenv("CAUTIOUS_DESCRIPTION_ENABLED", "true").lower() in ("true", "1", "yes"))
    DEVICE: str = Field(default_factory=lambda: os.getenv("DEVICE", "cpu"))

    # Image & Processing Constraints
    MAX_IMAGE_SIZE_MB: float = Field(default_factory=lambda: float(os.getenv("MAX_IMAGE_SIZE_MB", "5.0")))
    MAX_IMAGE_SIZE_BYTES: int = 0
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    MAX_CONCURRENT_REQUESTS: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")))
    INFERENCE_TIMEOUT_SECONDS: float = Field(default_factory=lambda: float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "15.0")))

    # Security
    API_KEY: str = Field(default_factory=lambda: os.getenv("API_KEY", ""))

    # Text-to-Speech Audio Configuration
    # Options for TTS_VOICE:
    # - 'female-north' or 'gtts' (Giọng nữ miền Bắc - Chị Google chuẩn)
    # - 'vi-VN-NamMinhNeural' (Giọng nam miền Bắc - Microsoft Edge)
    # - 'vi-VN-HoaiMyNeural' (Giọng nữ miền Nam - Microsoft Edge)
    TTS_ENGINE: str = Field(default_factory=lambda: os.getenv("TTS_ENGINE", "gtts"))
    TTS_VOICE: str = Field(default_factory=lambda: os.getenv("TTS_VOICE", "female-north"))
    TTS_RATE: str = Field(default_factory=lambda: os.getenv("TTS_RATE", "+0%"))
    AUDIO_TTL_MINUTES: int = Field(default_factory=lambda: int(os.getenv("AUDIO_TTL_MINUTES", "30")))
    MAX_CONCURRENT_TTS: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_TTS", "3")))
    TTS_TIMEOUT_SECONDS: float = Field(default_factory=lambda: float(os.getenv("TTS_TIMEOUT_SECONDS", "12.0")))

    # Object Prioritization Settings (Assistive Navigation & Scene Description)
    # Note: These weights prioritize relevant navigation cues for voice feedback;
    # they do NOT estimate absolute physical meters and do NOT replace HC-SR04 ultrasonic sensors.
    PRIORITY_WEIGHT_BOTTOM: float = Field(default_factory=lambda: float(os.getenv("PRIORITY_WEIGHT_BOTTOM", "0.35")))
    PRIORITY_WEIGHT_AREA: float = Field(default_factory=lambda: float(os.getenv("PRIORITY_WEIGHT_AREA", "0.20")))
    PRIORITY_WEIGHT_CENTER: float = Field(default_factory=lambda: float(os.getenv("PRIORITY_WEIGHT_CENTER", "0.20")))
    PRIORITY_WEIGHT_CONFIDENCE: float = Field(default_factory=lambda: float(os.getenv("PRIORITY_WEIGHT_CONFIDENCE", "0.15")))
    PRIORITY_WEIGHT_CORRIDOR: float = Field(default_factory=lambda: float(os.getenv("PRIORITY_WEIGHT_CORRIDOR", "0.10")))
    PRIORITY_EDGE_PENALTY: float = Field(default_factory=lambda: float(os.getenv("PRIORITY_EDGE_PENALTY", "0.55")))
    PRIORITY_MIN_THRESHOLD: float = Field(default_factory=lambda: float(os.getenv("PRIORITY_MIN_THRESHOLD", "0.15")))
    MAX_SPOKEN_OBJECTS: int = Field(default_factory=lambda: int(os.getenv("MAX_SPOKEN_OBJECTS", "2")))

    # Dynamic Object Selection Settings
    # Controls whether to describe a 2nd object based on absolute/relative score & marginal information value
    DYNAMIC_SELECTION_ENABLED: bool = Field(default_factory=lambda: os.getenv("DYNAMIC_SELECTION_ENABLED", "true").lower() in ("true", "1", "yes"))
    SECONDARY_MIN_SCORE: float = Field(default_factory=lambda: float(os.getenv("SECONDARY_MIN_SCORE", "0.35")))
    SECONDARY_RELATIVE_RATIO: float = Field(default_factory=lambda: float(os.getenv("SECONDARY_RELATIVE_RATIO", "0.55")))
    SAME_CLASS_RELATIVE_RATIO: float = Field(default_factory=lambda: float(os.getenv("SAME_CLASS_RELATIVE_RATIO", "0.70")))
    URGENT_GROUND_HAZARD_Y2: float = Field(default_factory=lambda: float(os.getenv("URGENT_GROUND_HAZARD_Y2", "0.70")))

    # Project & Model Directories
    BASE_DIR: Path = PROJECT_ROOT
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    STATIC_DIR: Path = PROJECT_ROOT / "static"

    # Storage Paths for ESP32 Real Data (Unified under data/)
    # - data/images: Lưu trữ ảnh chụp thực tế từ ESP32-CAM (kèm Bounding Box)
    # - data/audio : Lưu trữ các file giọng nói MP3 tương ứng gửi cho ESP32-CAM
    DATA_DIR: Path = PROJECT_ROOT / "data"
    IMAGE_DIR: Path = PROJECT_ROOT / "data" / "images"
    AUDIO_DIR: Path = PROJECT_ROOT / "data" / "audio"

    # Backward compatibility aliases
    DEBUG_DIR: Path = PROJECT_ROOT / "data" / "images"
    TEMP_AUDIO_DIR: Path = PROJECT_ROOT / "data" / "audio"
    SAVE_DEBUG_IMAGES: bool = Field(default_factory=lambda: os.getenv("SAVE_DEBUG_IMAGES", "true").lower() in ("true", "1", "yes"))

    def get_audio_url(self, audio_id: str) -> str:
        """Construct full public LAN URL for an audio file."""
        return f"http://{self.LOCAL_IP}:{self.PORT}/api/audio/{audio_id}"

    def model_post_init(self, __context):
        self.MAX_IMAGE_SIZE_BYTES = int(self.MAX_IMAGE_SIZE_MB * 1024 * 1024)

    def resolve_model_path(self) -> Path:
        """Resolve model path, checking models/ directory and fallback locations."""
        model_str = self.YOLO_MODEL_PATH or self.MODEL_PATH or "models/yolov8n.pt"
        p = Path(model_str)
        if p.is_absolute() and p.exists():
            return p
        # Check relative to BASE_DIR
        candidate1 = self.BASE_DIR / model_str
        if candidate1.exists():
            return candidate1
        # Check in models/
        candidate2 = self.MODELS_DIR / Path(model_str).name
        if candidate2.exists():
            return candidate2
        # Fallback to direct name in BASE_DIR
        candidate3 = self.BASE_DIR / Path(model_str).name
        if candidate3.exists():
            return candidate3
        # Default to candidate2 or candidate1
        return candidate2 if candidate2.parent.exists() else candidate1


settings = Settings()
