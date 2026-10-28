"""
Description Service for Assistive AI Server.
Generates concise, natural, and safe Vietnamese scene descriptions for visually impaired users.

10 Core Design Rules:
1. Prioritizes the most relevant objects (highest priority score).
2. Describes at most 2 objects or semantic groups.
3. Keeps sentences concise and natural (typically 8–20 words).
4. Uses natural spatial directions:
   - "Phía trước..."
   - "Phía trước bên trái..."
   - "Phía trước bên phải..."
5. Excludes distant objects from spoken output to avoid audio clutter.
6. Prevents word repetitions and extra punctuation/commas.
7. Does NOT infer exact metric distance from bounding boxes or assert "ở xa hơn".
8. Does NOT claim objects are approaching, moving, or blocking the path from static images.
9. Does NOT assert "đường đi an toàn" when no objects are detected.
10. Uses cautious generic descriptions ("vật thể") when confidence is low/uncertain.
"""
from collections import Counter
from typing import List, Optional
from app.schemas.response import DetectedObject, PositionEnum
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
        """
        Create a concise Vietnamese description following the 10 guidelines.
        """
        # Rule 9: If no groups selected
        if not selected_groups:
            if total_objects_count == 0:
                return "Không phát hiện vật thể trong khung hình."
            return "Không phát hiện vật thể quan trọng phía trước."

        # Rule 2: Limit to at most 2 groups
        groups = selected_groups[:2]

        # Case A: Exactly 1 group
        if len(groups) == 1:
            g = groups[0]
            phrase = format_count_phrase(1, g.label_vi)
            prefix = "" if phrase.startswith(("nghi là ", "có thể là ")) else "có "
            if g.position == PositionEnum.CENTER:
                return f"Phía trước {prefix}{phrase}."
            elif g.position == PositionEnum.LEFT:
                return f"Phía trước bên trái {prefix}{phrase}."
            else:
                return f"Phía trước bên phải {prefix}{phrase}."

        # Case B: Exactly 2 groups
        g1, g2 = groups[0], groups[1]
        lbl1 = g1.label_vi
        lbl2 = g2.label_vi

        # Subcase B1: Same spatial position
        if g1.position == g2.position:
            if lbl1 == lbl2:
                phrase = format_count_phrase(2, lbl1)
                if g1.position == PositionEnum.CENTER:
                    return f"Phía trước có {phrase}."
                elif g1.position == PositionEnum.LEFT:
                    return f"Phía trước bên trái có {phrase}."
                else:
                    return f"Phía trước bên phải có {phrase}."
            else:
                # Prioritize mentioning person first if one is a person
                if "người" in lbl2 and "người" not in lbl1:
                    lbl1, lbl2 = lbl2, lbl1

                phrase1 = format_count_phrase(1, lbl1)
                phrase2 = format_count_phrase(1, lbl2)

                if g1.position == PositionEnum.CENTER:
                    return f"Phía trước có {phrase1} và {phrase2}."
                elif g1.position == PositionEnum.LEFT:
                    return f"Phía trước bên trái có {phrase1} và {phrase2}."
                else:
                    return f"Phía trước bên phải có {phrase1} và {phrase2}."

        # Subcase B2: Different spatial positions
        # Standardize ordering: Center first, or Left before Right for scanning consistency
        if g2.position == PositionEnum.CENTER and g1.position != PositionEnum.CENTER:
            g1, g2 = g2, g1
            lbl1, lbl2 = g1.label_vi, g2.label_vi
        elif g1.position == PositionEnum.RIGHT and g2.position == PositionEnum.LEFT:
            g1, g2 = g2, g1
            lbl1, lbl2 = g1.label_vi, g2.label_vi

        phrase1 = format_count_phrase(1, lbl1)
        phrase2 = format_count_phrase(1, lbl2)

        # First clause
        if g1.position == PositionEnum.CENTER:
            part1 = f"Phía trước có {phrase1}"
        elif g1.position == PositionEnum.LEFT:
            part1 = f"Phía trước bên trái có {phrase1}"
        else:
            part1 = f"Phía trước bên phải có {phrase1}"

        # Second clause
        if g2.position == PositionEnum.CENTER:
            part2 = f"phía giữa có {phrase2}"
        elif g2.position == PositionEnum.LEFT:
            part2 = f"bên trái có {phrase2}"
        else:
            part2 = f"bên phải có {phrase2}"

        # Clean formatting
        sentence = f"{part1}, {part2}."
        sentence = sentence.replace("  ", " ").replace(", ,", ",")
        return sentence


# Global singleton instance
description_service = DescriptionService()
