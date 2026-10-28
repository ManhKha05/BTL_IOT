"""
Object Prioritization Service for Assistive AI Server.
Evaluates detected object relevance for visually impaired walking assistance:
- Combines multi-feature heuristic scoring: confidence, area ratio, perspective bottom position,
  horizontal corridor alignment, object class relevance, and edge clipping penalty.
- Configurable weights via Settings.
- Implements Dynamic Object Selection:
  * Selects 1 or 2 entities based on absolute relevance, relative score ratio, and marginal information gain.
  * Does NOT force selecting 2 objects if the 2nd object is insignificant or redundant.
  * Differentiates same-class redundancy vs distinct hazard classes.
  * Preserves urgent low-lying ground obstacles in the walking corridor.
  * Suppresses distant horizon background noise.
"""
from typing import Dict, List, Optional, Set, Tuple
from app.core.config import settings
from app.schemas.response import DetectedObject, PositionEnum

# Hazard and navigation relevance weights for pedestrian walking assistance:
# - High relevance (1.0): people, vehicles, tripping hazards, stairs, doors
# - Moderate relevance (0.75 - 0.90): animals, larger furniture, poles
# - Lower relevance (0.40 - 0.50): handheld/small static items (unless right in front)
CLASS_HAZARD_WEIGHTS: Dict[str, float] = {
    # High priority: Moving entities & direct travel hazards
    "person": 1.0,
    "bicycle": 1.0,
    "car": 1.0,
    "motorcycle": 1.0,
    "bus": 1.0,
    "truck": 1.0,
    "train": 1.0,
    "dog": 1.0,
    "horse": 0.95,
    "cow": 0.95,
    
    # Path obstacles & navigation landmarks
    "stairs": 1.0,
    "staircase": 1.0,
    "steps": 0.95,
    "door": 1.0,
    "pole": 1.0,
    "utility pole": 1.0,
    "traffic light": 0.95,
    "stop sign": 0.95,
    "fire hydrant": 0.95,
    "crosswalk": 1.0,
    "road barrier": 1.0,
    "trash bin": 0.95,
    "garbage bin": 0.95,
    "bollard": 0.95,
    "traffic cone": 0.90,
    "curb": 0.85,
    "flower pot": 0.80,
    "chair": 0.90,  # tripping obstacle on sidewalk/indoor
    "bench": 0.85,
    
    # Animals & larger furniture
    "cat": 0.85,
    "couch": 0.80,
    "bed": 0.75,
    "dining table": 0.75,
    "potted plant": 0.75,
    "toilet": 0.75,
    
    # Handheld / small static accessories (less critical for walking)
    "backpack": 0.50,
    "umbrella": 0.55,
    "handbag": 0.50,
    "suitcase": 0.70,
    "bottle": 0.45,
    "cup": 0.45,
    "cell phone": 0.40,
    "laptop": 0.45,
    "tv": 0.60,
    "book": 0.40,
}


def calculate_priority_score(
    bbox: List[int],
    label: str,
    img_width: int,
    img_height: int,
    confidence: float = 0.80
) -> float:
    """
    Calculate object priority score in [0.0, 1.0] for scene description.
    
    LƯU Ý AN TOÀN:
    Điểm số này phục vụ lựa chọn vật thể nổi bật nhất để tạo câu mô tả ngữ cảnh trực quan (scene description)
    bằng giọng nói, KHÔNG coi bounding box/y2 là khoảng cách vật lý tuyệt đối và KHÔNG thay thế
    cảm biến siêu âm HC-SR04 đo khoảng cách thực tế để cảnh báo va chạm.
    
    Features:
    1. Ground proximity (Perspective y2/H): Objects near or above horizon (y2 <= 0.40H) have minimal
       ground contact value; objects extending toward the bottom (y2 -> H) receive high ground value.
    2. Area ratio: Bounding box area relative to frame size, normalized to 15% frame threshold.
       (Note: Area is NOT used as a hard rejection cutoff; small obstacles in path remain detectable).
    3. Horizontal position: Normalized closeness to central forward walking axis.
    4. Central walking corridor: Bonus for presence in central 33%-67% horizontal travel band.
    5. Confidence: Detection reliability score.
    6. Object class relevance: Moving entities and travel hazards rank higher than handheld items.
    7. Edge clipping penalty: Objects cut off at image margins receive a discount.
    """
    x1, y1, x2, y2 = bbox
    w = max(1, img_width)
    h = max(1, img_height)

    # 1. Perspective ground contact feature
    ground_ratio = max(0.0, min(1.0, y2 / float(h)))
    if ground_ratio <= 0.40:
        s_bottom = 0.20 * (ground_ratio / 0.40)
    else:
        s_bottom = 0.20 + 0.80 * (((ground_ratio - 0.40) / 0.60) ** 1.5)
    s_bottom = max(0.0, min(1.0, s_bottom))

    # 2. Area ratio feature (normalized so >= 15% frame area reaches saturation 1.0)
    box_area = float(max(0, x2 - x1) * max(0, y2 - y1))
    frame_area = float(w * h)
    area_ratio = box_area / frame_area if frame_area > 0 else 0.0
    s_area = max(0.0, min(1.0, area_ratio / 0.15))

    # 3. Horizontal position / distance from center axis
    center_x = (x1 + x2) / 2.0
    d_center = abs(center_x - (w / 2.0)) / (w / 2.0)
    d_center = max(0.0, min(1.0, d_center))
    s_center = 1.0 - d_center

    # 4. Central walking corridor feature (bonus for being in direct travel path [33%, 67%])
    s_corridor = 1.0 if (w * 0.33 <= center_x <= w * 0.67) else 0.0

    # 5. Detection confidence feature
    s_conf = max(0.0, min(1.0, float(confidence)))

    # 6. Weighted combination of linear features
    wb = settings.PRIORITY_WEIGHT_BOTTOM
    wa = settings.PRIORITY_WEIGHT_AREA
    wc = settings.PRIORITY_WEIGHT_CENTER
    wf = getattr(settings, "PRIORITY_WEIGHT_CONFIDENCE", 0.15)
    wk = getattr(settings, "PRIORITY_WEIGHT_CORRIDOR", 0.10)

    base_score = (wb * s_bottom) + (wa * s_area) + (wc * s_center) + (wf * s_conf) + (wk * s_corridor)

    # 7. Class hazard / relevance factor
    class_weight = CLASS_HAZARD_WEIGHTS.get(label.lower(), 0.70)

    # 8. Edge clipping penalty (lateral edges: objects cut off entering/exiting field of view)
    edge_margin_x = max(3, int(w * 0.025))
    is_at_edge = (x1 <= edge_margin_x) or (x2 >= (w - edge_margin_x))
    edge_penalty = settings.PRIORITY_EDGE_PENALTY if is_at_edge else 1.0

    # Final combined score bounded to [0.0, 1.0]
    final_score = base_score * class_weight * edge_penalty
    return round(float(max(0.0, min(1.0, final_score))), 3)


def get_object_spoken_label(obj: DetectedObject) -> str:
    """
    Get spoken Vietnamese label for an object.
    If CAUTIOUS_DESCRIPTION_ENABLED is True and confidence < CONFIDENCE_CERTAIN_THRESHOLD:
    - For high-priority navigation & falling hazards (stairs, staircase, steps, road barrier, bollard):
      returns 'nghi là {label_vi}' (e.g. 'nghi là cầu thang') so the user is alerted to the hazard cautiously.
    - For general items: return generic 'vật thể' to prevent misleading false assertions.
    """
    if settings.CAUTIOUS_DESCRIPTION_ENABLED and obj.confidence < settings.CONFIDENCE_CERTAIN_THRESHOLD:
        hazard_classes = {"stairs", "staircase", "steps", "road barrier", "traffic cone", "bollard"}
        if obj.label.lower() in hazard_classes:
            return f"nghi là {obj.label_vi}"
        return "vật thể"
    return obj.label_vi


class ObjectPrioritizationService:
    """
    Service responsible for prioritizing detected objects:
    - Calculates priority scores for all detected objects.
    - Sets cautious spoken labels when confidence is uncertain.
    - Implements Dynamic Object Selection to choose 1 or 2 entities based on real marginal value.
    """

    def score_all_objects(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int
    ) -> List[DetectedObject]:
        """
        Calculates priority score and spoken label for each object in the list.
        Modifies objects in-place and sorts by priority score descending.
        """
        for obj in objects:
            conf = getattr(obj, "confidence", 0.80)
            score = calculate_priority_score(
                bbox=obj.bbox,
                label=obj.label,
                img_width=img_width,
                img_height=img_height,
                confidence=conf
            )
            obj.priority_score = score
            obj.is_spoken = False
            obj.spoken_label = get_object_spoken_label(obj)

        objects.sort(key=lambda o: (o.priority_score or 0.0, o.relative_size or 0.0), reverse=True)
        return objects

    def _extract_labels(self, candidate: any) -> Set[str]:
        """Extract set of lowercase English class labels from a candidate or semantic group."""
        members = getattr(candidate, "member_objects", None)
        if members:
            return {m.label.lower().strip() for m in members if hasattr(m, "label")}
        lbl = getattr(candidate, "label", None)
        if lbl:
            return {lbl.lower().strip()}
        return set()

    def select_top_candidates(
        self,
        candidates: List[any],
        img_width: int,
        img_height: int,
        max_spoken: Optional[int] = None,
        min_threshold: Optional[float] = None
    ) -> List[any]:
        """
        Dynamic Object Selection algorithm:
        1. Checks absolute minimum threshold for primary object.
        2. Does NOT blindly take 2 objects. Evaluates whether a 2nd object provides significant
           marginal information value.
        3. Applies relative dominance ratio (S2 / S1): if secondary object is heavily overshadowed,
           it is omitted to keep the sentence concise and focused.
        4. Applies same-class penalty: two objects of the same class (e.g. 2 cars) require a higher
           relative threshold (default >= 0.70) to prevent saying "1 ô tô phía trước, 1 ô tô bên phải"
           when the second car is just a background vehicle.
        5. Preserves urgent ground tripping hazards in the central walking corridor regardless of relative ratio.
        6. Suppresses distant horizon background specks.
        """
        limit = max_spoken if max_spoken is not None else settings.MAX_SPOKEN_OBJECTS
        thresh = min_threshold if min_threshold is not None else settings.PRIORITY_MIN_THRESHOLD

        # Filter candidates meeting basic minimum score
        valid = [c for c in candidates if (getattr(c, "priority_score", 0.0) or 0.0) >= thresh]
        if not valid:
            return []

        primary = valid[0]
        s1 = getattr(primary, "priority_score", 0.0) or 0.0
        selected = [primary]

        # If only 1 object allowed or only 1 valid candidate
        if limit <= 1 or len(valid) == 1:
            return selected

        # If Dynamic Selection is disabled in configuration, fallback to top-N
        if not getattr(settings, "DYNAMIC_SELECTION_ENABLED", True):
            return valid[:limit]

        W = max(1, img_width)
        H = max(1, img_height)

        p_bbox = getattr(primary, "bbox", [0, 0, 0, 0])
        p_y2 = p_bbox[3] / float(H)
        p_area = float(max(0, p_bbox[2] - p_bbox[0]) * max(0, p_bbox[3] - p_bbox[1])) / float(W * H)
        is_prominent_foreground = (p_y2 >= 0.70) or (p_area >= 0.12)
        primary_labels = self._extract_labels(primary)

        secondary_min_score = getattr(settings, "SECONDARY_MIN_SCORE", 0.35)
        secondary_rel_ratio = getattr(settings, "SECONDARY_RELATIVE_RATIO", 0.55)
        same_class_rel_ratio = getattr(settings, "SAME_CLASS_RELATIVE_RATIO", 0.70)
        urgent_hazard_y2 = getattr(settings, "URGENT_GROUND_HAZARD_Y2", 0.70)

        for cand in valid[1:]:
            s2 = getattr(cand, "priority_score", 0.0) or 0.0
            c_bbox = getattr(cand, "bbox", [0, 0, 0, 0])
            c_cx = (c_bbox[0] + c_bbox[2]) / 2.0
            c_y2 = c_bbox[3] / float(H)
            c_area = float(max(0, c_bbox[2] - c_bbox[0]) * max(0, c_bbox[3] - c_bbox[1])) / float(W * H)
            cand_labels = self._extract_labels(cand)

            # Check if this candidate is an urgent ground obstacle directly in the walking corridor
            is_in_corridor = (W * 0.33 <= c_cx <= W * 0.67)
            is_ground_obstacle = (c_y2 >= urgent_hazard_y2) and is_in_corridor
            # Obstacle categories with tripping risk
            tripping_classes = {"chair", "dog", "cat", "stairs", "pole", "stop sign", "fire hydrant", "suitcase"}
            is_tripping_hazard = bool(cand_labels & tripping_classes)

            # Exception: An urgent ground-level path obstacle directly in the walking path
            # is always retained to protect the wearer from collision / tripping
            if is_ground_obstacle and (is_tripping_hazard or s2 >= 0.30):
                selected.append(cand)
                break

            # 1. Distant background speck suppression when foreground is prominent
            if is_prominent_foreground and (c_y2 <= 0.45 and c_area <= 0.03):
                continue

            # 2. Check absolute minimum score for secondary candidate
            if s2 < secondary_min_score:
                continue

            # 3. Check relative dominance ratio (S2 / S1)
            rel_ratio = s2 / s1 if s1 > 0 else 0.0
            is_same_class = bool(primary_labels & cand_labels)

            required_ratio = same_class_rel_ratio if is_same_class else secondary_rel_ratio
            if rel_ratio < required_ratio:
                # Secondary object is significantly overshadowed by primary object
                continue

            # Passed all filters -> Secondary object provides significant marginal value
            selected.append(cand)
            break

        return selected


# Global singleton instance
prioritization_service = ObjectPrioritizationService()
