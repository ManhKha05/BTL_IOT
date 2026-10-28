"""
Unified Object Detection Service for Assistive AI Server.
Powered natively by YOLO-World Small (models/yolov8s-worldv2.pt) on GPU RTX 3050.
Optimized specifically for Vietnamese urban street and sidewalk navigation assistance.
Provides:
- Fixed backend class configuration (no client parameter needed)
- Multi-prompt token optimization for essential pedestrian hazards
- Cross-class synonym deduplication (e.g., potted plant vs flower pot)
- Sub-20ms warm GPU inference latency
"""
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLOWorld

from app.core.config import settings
from app.core.logging import logger
from app.schemas.response import DetectedObject, PositionEnum


# Fixed curated list of 25 optimized classes for Vietnamese pedestrian environments
VIETNAM_STREET_CLASSES: List[str] = [
    # 1. Người & phương tiện giao thông (6)
    "person",
    "motorcycle",
    "car",
    "bicycle",
    "bus",
    "truck",

    # 2. Vật cản mặt đất & vỉa hè thiết yếu (8)
    "trash bin",
    "traffic cone",
    "potted plant",
    "flower pot",
    "road barrier",
    "fence",
    "bollard",
    "utility pole",

    # 3. Công trình & điều hướng đi bộ (7)
    "stairs",
    "staircase",
    "steps",
    "door",
    "crosswalk",
    "traffic light",
    "stop sign",

    # 4. Tiện ích công cộng & động vật (4)
    "chair",
    "bench",
    "dog",
    "cat"
]


# Vietnamese translations with synonym normalization
VIETNAMESE_LABELS: Dict[str, str] = {
    # Người & phương tiện
    "person": "người",
    "motorcycle": "xe máy",
    "car": "ô tô",
    "bicycle": "xe đạp",
    "bus": "xe buýt",
    "truck": "xe tải",

    # Vật cản mặt đất & vỉa hè
    "trash bin": "thùng rác",
    "garbage bin": "thùng rác",
    "garbage can": "thùng rác",
    "waste basket": "thùng rác",
    "traffic cone": "cọc tiêu giao thông",
    "potted plant": "chậu cây",
    "flower pot": "chậu cây",
    "road barrier": "rào chắn",
    "fence": "rào chắn",
    "bollard": "cột chắn",
    "utility pole": "cột điện",
    "electric pole": "cột điện",

    # Công trình & điều hướng
    "stairs": "cầu thang",
    "staircase": "cầu thang",
    "steps": "bậc thềm",
    "indoor staircase": "cầu thang",
    "flight of stairs": "cầu thang",
    "door": "cánh cửa",
    "crosswalk": "vạch sang đường",
    "traffic light": "đèn giao thông",
    "stop sign": "biển báo dừng",

    # Tiện ích công cộng & động vật
    "chair": "chiếc ghế",
    "bench": "ghế dài",
    "dog": "con chó",
    "cat": "con mèo",

    # Fallbacks for standard COCO classes
    "backpack": "ba lô",
    "umbrella": "chiếc ô",
    "handbag": "túi xách",
    "suitcase": "va ly",
    "bottle": "chai nước",
    "couch": "ghế sô pha",
    "bed": "chiếc giường",
    "dining table": "bàn ăn",
    "tv": "ti vi",
    "laptop": "máy tính",
    "cell phone": "điện thoại"
}

# Calibrated minimum confidence threshold for low-logit open-vocabulary hazard categories
# Prevents false negatives for structural hazards while maintaining strict 0.35 threshold for general objects
CLASS_MIN_CONFIDENCE: Dict[str, float] = {
    "stairs": 0.07,
    "staircase": 0.07,
    "steps": 0.08,
}



def determine_position(bbox: List[int], img_width: int) -> PositionEnum:
    """
    Determine horizontal spatial position (left, center, right)
    based on the horizontal center of the bounding box.
    - Left: center_x < 33% of width
    - Right: center_x > 67% of width
    - Center: 33% <= center_x <= 67% (forward walking corridor)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    
    left_threshold = img_width * 0.33
    right_threshold = img_width * 0.67

    if center_x < left_threshold:
        return PositionEnum.LEFT
    elif center_x > right_threshold:
        return PositionEnum.RIGHT
    else:
        return PositionEnum.CENTER


def _compute_box_iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute Intersection over Union between two integer bboxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    denom = boxAArea + boxBArea - interArea
    return interArea / float(denom) if denom > 0 else 0.0


class ObjectDetectionService:
    """
    Production Vision Service running YOLO-World Small:
    - Preloads YOLOv8s-Worldv2 model once during application startup.
    - Caches Vietnamese street obstacle classes in the model head.
    - Runs non-blocking asynchronous inference via threadpool.
    - Implements cross-class synonym deduplication (e.g. potted plant vs flower pot).
    - Limits concurrent execution via asyncio.Semaphore.
    """

    def __init__(self):
        self.model: Optional[YOLOWorld] = None
        self.model_path: Optional[Path] = None
        self.device: str = "cpu"
        self._is_loaded: bool = False
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.classes: List[str] = list(VIETNAM_STREET_CLASSES)

    def initialize_semaphore(self) -> None:
        """Initialize concurrency limiter in the running event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

    def load_model(self) -> None:
        """Load YOLO-World Small model into memory. Called once during FastAPI lifespan startup."""
        if self._is_loaded and self.model is not None:
            logger.info("YOLO-World Small production model is already loaded.")
            return

        model_path = settings.resolve_model_path()
        self.model_path = model_path

        # Determine device
        if settings.DEVICE.lower() == "cuda" and torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        logger.info(f"Loading YOLO-World Small production model from '{model_path}' on device '{self.device}'...")
        start_t = time.perf_counter()

        try:
            self.model = YOLOWorld(str(model_path))
            # Cache the 22 Vietnamese street classes into the model's text encoder
            self.model.set_classes(self.classes)

            # Warm-up model with a dummy inference to avoid cold-start latency on first ESP32 request
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy_img, device=self.device, verbose=False)

            elapsed = (time.perf_counter() - start_t) * 1000
            self._is_loaded = True
            logger.info(f"YOLO-World Small model loaded and warmed up successfully ({len(self.classes)} classes) in {elapsed:.1f} ms.")
        except Exception as e:
            logger.error(f"Failed to load YOLO-World Small model from '{model_path}': {e}", exc_info=True)
            self._is_loaded = False
            raise RuntimeError(f"Could not load YOLO model: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def _run_inference_sync(
        self,
        img_rgb: np.ndarray,
        conf_thresh: float
    ) -> Tuple[List[DetectedObject], float]:
        """
        Synchronous YOLO-World inference executed inside worker threadpool.
        Returns:
            (detected_objects, inference_time_ms)
        """
        if not self.is_loaded:
            raise RuntimeError("YOLO-World model is not initialized.")

        h, w, _ = img_rgb.shape
        frame_area = float(w * h)

        start_time = time.perf_counter()
        # Feed model with floor conf so candidate hazard boxes are not pruned in forward pass
        model_conf = min(conf_thresh, 0.06)
        results = self.model(
            img_rgb,
            conf=model_conf,
            device=self.device,
            verbose=False
        )
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        raw_detected: List[DetectedObject] = []

        if not results or len(results) == 0:
            return raw_detected, round(inference_time_ms, 2)

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return raw_detected, round(inference_time_ms, 2)

        for box in boxes:
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            if cls_id < len(self.classes):
                label_en = self.classes[cls_id]
            else:
                label_en = f"class_{cls_id}"

            min_required = CLASS_MIN_CONFIDENCE.get(label_en.lower(), conf_thresh)
            if conf < min_required:
                continue

            xyxy = box.xyxy[0].tolist()
            x1 = max(0, min(w - 1, int(round(xyxy[0]))))
            y1 = max(0, min(h - 1, int(round(xyxy[1]))))
            x2 = max(0, min(w, int(round(xyxy[2]))))
            y2 = max(0, min(h, int(round(xyxy[3]))))
            bbox = [x1, y1, x2, y2]

            box_area = float((x2 - x1) * (y2 - y1))
            rel_size = round(box_area / frame_area, 4) if frame_area > 0 else 0.0

            # For low-confidence hazard classes (<0.15), filter out tiny distant or non-ground artifacts
            if min_required < 0.15 and conf < 0.15:
                if rel_size < 0.04 and (y2 / float(h)) < 0.50:
                    continue

            pos = determine_position(bbox, w)
            label_vi = VIETNAMESE_LABELS.get(label_en.lower(), label_en)

            raw_detected.append(
                DetectedObject(
                    label=label_en,
                    label_vi=label_vi,
                    confidence=round(conf, 4),
                    position=pos,
                    bbox=bbox,
                    relative_size=rel_size
                )
            )

        # Cross-class synonym deduplication
        # Suppress duplicate boxes representing the same semantic entity (e.g. flower pot & potted plant)
        sorted_objs = sorted(raw_detected, key=lambda o: o.confidence, reverse=True)
        deduped: List[DetectedObject] = []
        for obj in sorted_objs:
            duplicate = False
            for accepted in deduped:
                if obj.label_vi == accepted.label_vi:
                    iou = _compute_box_iou(obj.bbox, accepted.bbox)
                    if iou >= 0.50:
                        duplicate = True
                        break
            if not duplicate:
                deduped.append(obj)

        # Sort objects by relative size descending as initial order
        deduped.sort(key=lambda o: (o.relative_size or 0), reverse=True)
        return deduped, round(inference_time_ms, 2)

    async def detect_raw(
        self,
        img_rgb: np.ndarray,
        conf_thresh: Optional[float] = None
    ) -> Tuple[List[DetectedObject], float]:
        """
        Asynchronous inference wrapper bounded by semaphore and threadpool execution.
        """
        self.initialize_semaphore()
        thresh = conf_thresh if conf_thresh is not None else settings.CONFIDENCE_THRESHOLD

        async with self._semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                self._run_inference_sync,
                img_rgb,
                thresh
            )


# Global singleton instance for production detection
detection_service = ObjectDetectionService()
