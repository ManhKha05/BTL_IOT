"""Turns selected objects and their positions into concise Vietnamese speech."""
from collections import Counter
from typing import List
from app.schemas.response import PositionEnum
from app.services.object_relationship_service import SemanticGroup

NUM_WORDS_VI = {
    1: "một",
    2: "hai",
    3: "ba",
    4: "bốn",
    5: "năm",
    6: "sáu",
    7: "bảy",
    8: "tám",
    9: "chín",
    10: "mười",
}

POSITION_PHRASES = {
    PositionEnum.CENTER: "Ở giữa",
    PositionEnum.LEFT: "Bên trái",
    PositionEnum.RIGHT: "Bên phải",
}


def format_count_phrase(count: int, label_vi: str) -> str:
    """
    Format object quantity and label in natural Vietnamese:
    - count=1, 'người' -> 'một người'
    - count=1, 'con chó' -> 'một con chó'
    - count=2, 'người' -> 'hai người'
    - count=2, 'người đang đi xe máy' -> 'hai người đang đi xe máy'
    """
    cleaned_label = label_vi.strip()
    num_str = NUM_WORDS_VI.get(count, str(count))

    # If count is 1
    if count == 1:
        # Check if already starts with a numeral or uncertainty phrase
        if cleaned_label.startswith(("một ", "hai ", "ba ", "nghi là ", "có thể là ")):
            return cleaned_label
        return f"một {cleaned_label}"

    # If count > 1
    # Check if starts with singular classifiers like "một chiếc", "chiếc", "cái"
    if cleaned_label.startswith("chiếc "):
        cleaned_label = cleaned_label[6:]
    elif cleaned_label.startswith("cái "):
        cleaned_label = cleaned_label[4:]

    return f"{num_str} {cleaned_label}"


class DescriptionService:
    """Service producing concise, natural Vietnamese descriptions tailored for assistive audio."""

    def generate_description(
        self,
        selected_groups: List[SemanticGroup],
        total_objects_count: int = 0
    ) -> str:
        if not selected_groups:
            if total_objects_count == 0:
                return "Chưa nhận diện rõ trong khung hình."
            return "Chưa nhận diện rõ điều gì đáng chú ý."

        groups = selected_groups[:2]
        sentences = []
        for position, direction in POSITION_PHRASES.items():
            labels = Counter(group.label_vi for group in groups if group.position == position)
            for label, count in labels.items():
                object_name = label.removeprefix("nghi là ").removeprefix("có thể là ")
                phrase = format_count_phrase(count, object_name)
                verb = "có thể có" if object_name != label else "có"
                sentences.append(f"{direction} {verb} {phrase}.")
        return " ".join(sentences)


# Global singleton instance
description_service = DescriptionService()
