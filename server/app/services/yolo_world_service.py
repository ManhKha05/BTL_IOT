"""
YOLO-World Service compatibility module.
Delegates directly to the unified production YOLOService (which runs YOLO-World Small
with curated Vietnam street obstacle classes).
Guarantees zero duplicate model preloading and 100% backward compatibility for benchmark scripts and tests.
"""
from typing import Dict, List
from app.services.object_detection_service import (
    VIETNAM_STREET_CLASSES,
    VIETNAMESE_LABELS,
    detection_service,
)
from app.services.yolo_service import YOLOService, yolo_service

# Class list and labels matching unified Vietnamese street obstacle configuration
YOLO_WORLD_CLASSES: List[str] = list(VIETNAM_STREET_CLASSES)
YOLO_WORLD_LABELS: Dict[str, str] = dict(VIETNAMESE_LABELS)

YOLO_WORLD_CLASSES_BY_GROUP: Dict[str, List[str]] = {
    "person_vehicle": [
        "person",
        "car",
        "motorcycle",
        "bicycle",
        "bus",
        "truck"
    ],
    "ground_obstacles": [
        "trash bin",
        "garbage bin",
        "utility pole",
        "bollard",
        "road barrier",
        "fence",
        "traffic cone",
        "flower pot",
        "potted plant",
        "chair",
        "bench"
    ],
    "infrastructure": [
        "stairs",
        "door",
        "curb",
        "crosswalk",
        "tree"
    ],
    "traffic_animals": [
        "traffic light",
        "stop sign",
        "dog",
        "cat"
    ]
}

# Backward compatibility alias
YOLOWorldService = YOLOService

# Unified singleton instance
yolo_world_service: YOLOService = yolo_service
