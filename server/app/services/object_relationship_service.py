"""
Object Relationship Service for Assistive AI Server.
Analyzes geometric and spatial relationships between objects:
- Prioritized relations:
  * person + motorcycle (riding vs standing near)
  * person + bicycle (riding vs standing near)
  * person + car (in-car/driving vs standing near)
- Applies Greedy Bipartite Matching to prevent multi-assignment.
- Groups reliable relations into semantic groups to prevent duplicate counting.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from app.core.config import settings
from app.schemas.response import DetectedObject, ObjectRelationship, PositionEnum
from app.services.object_detection_service import VIETNAMESE_LABELS, determine_position


@dataclass
class SemanticGroup:
    """Đại diện cho một thực thể ngữ nghĩa sau khi gom nhóm (ví dụ: người đang đi xe máy)."""
    label_vi: str
    position: PositionEnum
    bbox: List[int]
    priority_score: float
    confidence: float
    is_grouped: bool
    relation_type: Optional[str] = None
    member_objects: List[DetectedObject] = field(default_factory=list)


class PrioritizeResult(tuple):
    """
    Tuple chứa (all_objects, spoken_objects) để tương thích 100% unpack 2 phần tử,
    đồng thời cho phép truy cập .relationships, .semantic_groups, và .selected_groups.
    """
    def __new__(cls, all_objects, spoken_objects, relationships=None, semantic_groups=None, selected_groups=None):
        return super().__new__(cls, (all_objects, spoken_objects))

    def __init__(self, all_objects, spoken_objects, relationships=None, semantic_groups=None, selected_groups=None):
        self.all_objects = all_objects
        self.spoken_objects = spoken_objects
        self.relationships = relationships if relationships is not None else []
        self.semantic_groups = semantic_groups if semantic_groups is not None else []
        self.selected_groups = selected_groups if selected_groups is not None else []


def analyze_object_relationships(
    objects: List[DetectedObject],
    img_width: int,
    img_height: int
) -> List[ObjectRelationship]:
    """
    Phân tích quan hệ hình học và không gian giữa các vật thể sau khi YOLO nhận diện.
    Kiểm tra các cặp quan hệ ưu tiên:
    1. person + motorcycle
    2. person + bicycle
    3. person + car
    
    Quy tắc:
    - Không kết luận hành động chỉ vì 2 bounding box giao nhau (chồng lấn).
    - Xét vị trí tương đối, tỷ lệ diện tích, căn chỉnh ngang/dọc, và tiếp xúc mặt đất.
    - Phân biệt quan hệ chắc chắn (is_certain=True) với người đứng cạnh/gần xe (is_certain=False).
    - Áp dụng Greedy Bipartite Matching để tránh gán lặp 1 người cho nhiều xe hoặc ngược lại.
    """
    relationships: List[ObjectRelationship] = []
    if not objects or len(objects) < 2:
        return relationships

    W = max(1, img_width)
    H = max(1, img_height)

    persons: List[Tuple[int, DetectedObject]] = []
    two_wheelers: List[Tuple[int, DetectedObject]] = []  # motorcycle, bicycle
    cars: List[Tuple[int, DetectedObject]] = []          # car

    for idx, obj in enumerate(objects):
        lbl = obj.label.lower().strip()
        if lbl == "person":
            persons.append((idx, obj))
        elif lbl in ("motorcycle", "bicycle"):
            two_wheelers.append((idx, obj))
        elif lbl in ("car", "truck", "bus"):
            cars.append((idx, obj))

    if not persons or (not two_wheelers and not cars):
        return relationships

    candidate_certain: List[Tuple[float, int, int, ObjectRelationship]] = []
    uncertain_relations: List[ObjectRelationship] = []

    # -------------------------------------------------------------
    # 1. Xét quan hệ person + motorcycle / bicycle (riding geometry)
    # -------------------------------------------------------------
    for p_idx, p_obj in persons:
        px1, py1, px2, py2 = p_obj.bbox
        pw = max(1, px2 - px1)
        ph = max(1, py2 - py1)
        p_area = float(pw * ph)
        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0

        for v_idx, v_obj in two_wheelers:
            vx1, vy1, vx2, vy2 = v_obj.bbox
            vw = max(1, vx2 - vx1)
            vh = max(1, vy2 - vy1)
            v_area = float(vw * vh)
            vcx = (vx1 + vx2) / 2.0

            # Tính giao điểm (intersection)
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = float(iw * ih)
            union_area = p_area + v_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0

            p_overlap = inter_area / p_area if p_area > 0 else 0.0
            h_overlap = iw / float(pw) if pw > 0 else 0.0

            v_type = v_obj.label.lower().strip()
            v_name_vi = VIETNAMESE_LABELS.get(v_type, v_obj.label_vi)

            # Điều kiện hình học người đang điều khiển xe máy / xe đạp:
            # - Dọc: Đầu người nhô lên trên nóc xe, chân thò vào xe, không thõng qua bánh xe
            cond_v_top = py1 < (vy1 + 0.15 * vh)
            cond_v_bottom = (py2 > vy1 + 0.20 * vh) and (py2 <= vy2 + 0.25 * vh)

            # - Ngang: Tâm người nằm trong phạm vi xe, độ chồng ngang lớn
            cond_h_align = (vx1 - 0.25 * vw <= pcx <= vx2 + 0.25 * vw)
            cond_h_overlap = h_overlap >= 0.35

            # - Tỷ lệ diện tích tương thích
            scale_ratio = p_area / v_area if v_area > 0 else 0.0
            cond_scale = 0.12 <= scale_ratio <= 3.5

            # - Mức độ chồng lấn 2D
            cond_overlap = (p_overlap >= 0.20) or (iou >= 0.15)

            is_riding = cond_v_top and cond_v_bottom and cond_h_align and cond_h_overlap and cond_scale and cond_overlap

            if is_riding:
                geom_score = (
                    0.4 * min(1.0, p_overlap / 0.40)
                    + 0.3 * min(1.0, h_overlap / 0.60)
                    + 0.3 * (1.0 - min(1.0, abs(pcx - vcx) / float(max(pw, vw))))
                )
                det_score = (p_obj.confidence * v_obj.confidence) ** 0.5
                confidence = round(float(geom_score * det_score), 3)

                if confidence >= 0.45:
                    rel = ObjectRelationship(
                        subject_idx=p_idx,
                        subject_label="person",
                        relation=f"person_on_{v_type}",
                        target_idx=v_idx,
                        target_label=v_type,
                        confidence=confidence,
                        is_certain=True,
                        description_vi=f"người đang đi {v_name_vi}"
                    )
                    candidate_certain.append((confidence, p_idx, v_idx, rel))
                    continue

            # Nếu không đủ cơ sở khẳng định đang điều khiển, xét quan hệ đứng cạnh hoặc gần nhau
            gap_x = max(0, max(px1, vx1) - min(px2, vx2))
            similar_ground = abs(py2 - vy2) <= 0.35 * H
            if (inter_area > 0 or gap_x <= 0.25 * W) and similar_ground:
                conf = round(float(0.40 * (p_obj.confidence * v_obj.confidence) ** 0.5), 3)
                uncertain_relations.append(
                    ObjectRelationship(
                        subject_idx=p_idx,
                        subject_label="person",
                        relation=f"person_near_{v_type}",
                        target_idx=v_idx,
                        target_label=v_type,
                        confidence=conf,
                        is_certain=False,
                        description_vi=f"người cạnh {v_name_vi}"
                    )
                )

        # -------------------------------------------------------------
        # 2. Xét quan hệ person + car (người trong ô tô hoặc cạnh ô tô)
        # -------------------------------------------------------------
        for c_idx, c_obj in cars:
            cx1, cy1, cx2, cy2 = c_obj.bbox
            cw = max(1, cx2 - cx1)
            ch = max(1, cy2 - cy1)
            c_area = float(cw * ch)
            c_center_x = (cx1 + cx2) / 2.0

            # Giao điểm
            ix1 = max(px1, cx1)
            iy1 = max(py1, cy1)
            ix2 = min(px2, cx2)
            iy2 = min(py2, cy2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = float(iw * ih)
            p_overlap = inter_area / p_area if p_area > 0 else 0.0

            c_type = c_obj.label.lower().strip()
            c_name_vi = VIETNAMESE_LABELS.get(c_type, c_obj.label_vi)

            # Điều kiện người ngồi bên trong ô tô:
            # - Bounding box người nằm phần lớn bên trong ô tô (>= 65% diện tích)
            # - Đầu và chân người không nhô hẳn ra ngoài ô tô
            cond_inside = (
                p_overlap >= 0.65
                and py1 >= cy1 - 0.05 * ch
                and py2 <= cy2 - 0.08 * ch
                and cx1 <= pcx <= cx2
            )

            if cond_inside:
                det_score = (p_obj.confidence * c_obj.confidence) ** 0.5
                conf = round(float(0.85 * p_overlap * det_score), 3)
                if conf >= 0.45:
                    rel = ObjectRelationship(
                        subject_idx=p_idx,
                        subject_label="person",
                        relation=f"person_in_{c_type}",
                        target_idx=c_idx,
                        target_label=c_type,
                        confidence=conf,
                        is_certain=True,
                        description_vi=f"người trong {c_name_vi}"
                    )
                    candidate_certain.append((conf, p_idx, c_idx, rel))
                    continue

            # Điều kiện người đứng cạnh ô tô (neutral/near):
            gap_x = max(0, max(px1, cx1) - min(px2, cx2))
            similar_ground = abs(py2 - cy2) <= 0.35 * H
            if (inter_area > 0 or gap_x <= 0.20 * W) and similar_ground:
                conf = round(float(0.40 * (p_obj.confidence * c_obj.confidence) ** 0.5), 3)
                uncertain_relations.append(
                    ObjectRelationship(
                        subject_idx=p_idx,
                        subject_label="person",
                        relation=f"person_near_{c_type}",
                        target_idx=c_idx,
                        target_label=c_type,
                        confidence=conf,
                        is_certain=False,
                        description_vi=f"người cạnh {c_name_vi}"
                    )
                )

    # Greedy Bipartite Matching cho các quan hệ chắc chắn
    candidate_certain.sort(key=lambda item: item[0], reverse=True)
    matched_persons: Set[int] = set()
    matched_vehicles: Set[int] = set()

    for conf, p_idx, v_idx, rel in candidate_certain:
        if p_idx not in matched_persons and v_idx not in matched_vehicles:
            matched_persons.add(p_idx)
            matched_vehicles.add(v_idx)
            relationships.append(rel)

    # Thêm các quan hệ đứng cạnh (is_certain=False) cho các đối tượng chưa ghép chắc chắn
    for u_rel in uncertain_relations:
        if u_rel.subject_idx not in matched_persons and u_rel.target_idx not in matched_vehicles:
            relationships.append(u_rel)

    return relationships


def build_semantic_groups(
    objects: List[DetectedObject],
    relationships: List[ObjectRelationship],
    img_width: int,
    img_height: int
) -> List[SemanticGroup]:
    """
    Nhóm các vật thể có quan hệ chắc chắn (is_certain=True) thành một thực thể ngữ nghĩa chung.
    Các vật thể còn lại giữ nguyên thành thực thể độc lập.
    Đảm bảo:
    - Không đếm trùng người và xe đã được nhóm.
    - Tính bounding box bao quanh, vị trí không gian và điểm ưu tiên tổng hợp.
    """
    W = max(1, img_width)
    H = max(1, img_height)

    certain_rel_map: Dict[int, ObjectRelationship] = {}
    vehicle_in_certain: Dict[int, int] = {}

    for rel in relationships:
        if rel.is_certain:
            certain_rel_map[rel.subject_idx] = rel
            vehicle_in_certain[rel.target_idx] = rel.subject_idx

    semantic_groups: List[SemanticGroup] = []
    processed_indices: Set[int] = set()

    for idx, obj in enumerate(objects):
        if idx in processed_indices:
            continue

        if idx in certain_rel_map:
            rel = certain_rel_map[idx]
            v_idx = rel.target_idx
            v_obj = objects[v_idx]

            px1, py1, px2, py2 = obj.bbox
            vx1, vy1, vx2, vy2 = v_obj.bbox
            comb_bbox = [min(px1, vx1), min(py1, vy1), max(px2, vx2), max(py2, vy2)]
            pos = determine_position(comb_bbox, W)

            p_score = max(obj.priority_score or 0.0, v_obj.priority_score or 0.0)
            
            sg = SemanticGroup(
                label_vi=rel.description_vi or f"người đang đi {v_obj.label_vi}",
                position=pos,
                bbox=comb_bbox,
                priority_score=p_score,
                confidence=rel.confidence,
                is_grouped=True,
                relation_type=rel.relation,
                member_objects=[obj, v_obj]
            )
            semantic_groups.append(sg)
            processed_indices.add(idx)
            processed_indices.add(v_idx)

        elif idx in vehicle_in_certain:
            continue

        else:
            lbl_vi = getattr(obj, "spoken_label", None) or obj.label_vi
            sg = SemanticGroup(
                label_vi=lbl_vi,
                position=obj.position,
                bbox=obj.bbox,
                priority_score=obj.priority_score or 0.0,
                confidence=obj.confidence,
                is_grouped=False,
                relation_type=None,
                member_objects=[obj]
            )
            semantic_groups.append(sg)
            processed_indices.add(idx)

    semantic_groups.sort(key=lambda g: g.priority_score, reverse=True)
    return semantic_groups


class ObjectRelationshipService:
    """Service analyzing spatial object relationships and constructing semantic groups."""

    def analyze(
        self,
        objects: List[DetectedObject],
        img_width: int,
        img_height: int
    ) -> List[ObjectRelationship]:
        return analyze_object_relationships(objects, img_width, img_height)

    def group(
        self,
        objects: List[DetectedObject],
        relationships: List[ObjectRelationship],
        img_width: int,
        img_height: int
    ) -> List[SemanticGroup]:
        return build_semantic_groups(objects, relationships, img_width, img_height)


# Global singleton instance
relationship_service = ObjectRelationshipService()
