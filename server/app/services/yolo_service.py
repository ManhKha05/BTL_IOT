"""
Unified YOLO Service Facade for Assistive AI Server.
Coordinates modular services:
- ObjectDetectionService (YOLO inference and positioning)
- ObjectPrioritizationService (Multi-feature heuristic scoring)
- ObjectRelationshipService (Spatial analysis & semantic grouping)
- DescriptionService (Natural Vietnamese sentence generation)

Maintains 100% backward compatibility for existing endpoints, models, and test suites.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.core.config import settings
from app.schemas.response import DetectedObject, ObjectRelationship, PositionEnum

# Re-export core items from modular services
from app.services.object_detection_service import (
    VIETNAMESE_LABELS,
    ObjectDetectionService,
    detection_service,
    determine_position,
)
from app.services.object_prioritization_service import (
    CLASS_HAZARD_WEIGHTS,
    ObjectPrioritizationService,
    calculate_priority_score,
    get_object_spoken_label,
    prioritization_service,
)
from app.services.object_relationship_service import (
    ObjectRelationshipService,
    PrioritizeResult,
    SemanticGroup,
    analyze_object_relationships,
    build_semantic_groups,
    relationship_service,
)
from app.services.description_service import (
    DescriptionService,
    description_service,
    format_count_phrase,
)


def generate_vietnamese_description_from_groups(
    selected_groups: List[SemanticGroup],
    total_objects_count: int = 0
) -> str:
    """Wrapper function preserving legacy API signature."""
    return description_service.generate_description(selected_groups, total_objects_count)


def prioritize_objects(
    objects: List[DetectedObject],
    img_width: int,
    img_height: int,
    max_spoken: Optional[int] = None
) -> PrioritizeResult:
    """
    Applies Object Prioritization with Relationship Analysis:
    1. Scores all detected objects using multi-feature priority calculation.
    2. Analyzes spatial relationships (person+motorcycle, person+bicycle, person+car).
    3. Builds semantic groups to prevent duplicate counting.
    4. Selects top prioritized groups meeting threshold.
    5. Flags is_spoken=True on constituent objects.
    6. Returns PrioritizeResult (compatible with tuple unpacking).
    """
    limit = max_spoken if max_spoken is not None else settings.MAX_SPOKEN_OBJECTS
    min_thresh = settings.PRIORITY_MIN_THRESHOLD

    # 1. Score all objects
    prioritization_service.score_all_objects(objects, img_width, img_height)

    # 2. Analyze geometric relationships
    relationships = relationship_service.analyze(objects, img_width, img_height)

    # 3. Build unified semantic groups
    semantic_groups = relationship_service.group(objects, relationships, img_width, img_height)

    # 4. Filter semantic groups with foreground prominence & min threshold
    selected_groups = prioritization_service.select_top_candidates(
        candidates=semantic_groups,
        img_width=img_width,
        img_height=img_height,
        max_spoken=limit,
        min_threshold=min_thresh
    )

    # 5. Mark spoken objects
    for g in selected_groups:
        for m_obj in g.member_objects:
            m_obj.is_spoken = True

    spoken_objects = [o for o in objects if o.is_spoken]
    return PrioritizeResult(objects, spoken_objects, relationships, semantic_groups, selected_groups)


class YOLOService:
    """
    Unified facade coordinating object detection, prioritization,
    relationship analysis, and natural Vietnamese scene description.
    """

    def __init__(self):
        self.detection_service = detection_service
        self.prioritization_service = prioritization_service
        self.relationship_service = relationship_service
        self.description_service = description_service

    @property
    def model(self):
        return self.detection_service.model

    @property
    def model_path(self):
        return self.detection_service.model_path

    @property
    def device(self):
        return self.detection_service.device

    @property
    def is_loaded(self) -> bool:
        return self.detection_service.is_loaded

    def initialize_semaphore(self) -> None:
        self.detection_service.initialize_semaphore()

    def load_model(self) -> None:
        """Preloads YOLO model once into memory."""
        self.detection_service.load_model()

    def _determine_position(self, bbox: List[int], img_width: int) -> PositionEnum:
        return determine_position(bbox, img_width)

    def calculate_priority_score(
        self,
        bbox: List[int],
        label: str,
        img_width: int,
        img_height: int,
        confidence: float = 0.80
    ) -> float:
        return calculate_priority_score(bbox, label, img_width, img_height, confidence)

    def prioritize_objects(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int,
        max_spoken: Optional[int] = None
    ) -> PrioritizeResult:
        return prioritize_objects(objects, img_width, img_height, max_spoken)

    def generate_vietnamese_description(
        self,
        spoken_objects: List[DetectedObject],
        img_height: Optional[int] = None,
        img_width: Optional[int] = None,
        relationships: Optional[List[ObjectRelationship]] = None,
        semantic_groups: Optional[List[SemanticGroup]] = None
    ) -> str:
        """Generates concise, natural Vietnamese description."""
        if semantic_groups is not None:
            return self.description_service.generate_description(semantic_groups, total_objects_count=len(spoken_objects))

        if not spoken_objects:
            return "Không phát hiện vật thể trong khung hình."

        w = img_width or 640
        h = img_height or 480
        if relationships is None:
            relationships = self.relationship_service.analyze(spoken_objects, w, h)
        groups = self.relationship_service.group(spoken_objects, relationships, w, h)
        return self.description_service.generate_description(groups, total_objects_count=len(spoken_objects))

    async def detect(
        self,
        img_rgb: np.ndarray,
        conf_thresh: Optional[float] = None,
        max_spoken: Optional[int] = None
    ) -> Tuple[List[DetectedObject], List[DetectedObject], List[ObjectRelationship], str, float]:
        """
        Unified detection pipeline:
        1. YOLO inference
        2. Prioritization and spatial scoring
        3. Relationship analysis & semantic grouping
        4. Natural Vietnamese description generation
        """
        h, w, _ = img_rgb.shape
        raw_objects, inf_time = await self.detection_service.detect_raw(img_rgb, conf_thresh)
        prioritize_res = self.prioritize_objects(raw_objects, w, h, max_spoken=max_spoken)

        all_objects = prioritize_res.all_objects
        spoken_objects = prioritize_res.spoken_objects
        relationships = prioritize_res.relationships
        selected_groups = prioritize_res.selected_groups

        description = self.description_service.generate_description(
            selected_groups=selected_groups,
            total_objects_count=len(all_objects)
        )
        return all_objects, spoken_objects, relationships, description, inf_time


# Global singleton instance
yolo_service = YOLOService()
