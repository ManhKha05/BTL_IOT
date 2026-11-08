"""
OCR Service for Assistive AI Server (Chức năng 3: đọc chữ trên biển hiệu và nhãn đồ vật).

Pipeline:
1. Nhận dạng chữ: Gemini (nếu có GEMINI_API_KEY, xem gemini_ocr_service) hoặc EasyOCR (vi + en,
   ảnh được đưa về cạnh dài OCR_MAX_IMAGE_SIDE). Gemini lỗi thì tự quay về EasyOCR.
2. Bỏ chữ ngắn bị cắt ở mép ảnh (thường của biển bên cạnh).
3. Gom các đoạn thành dòng (cùng cỡ chữ, cùng hàng); tách dòng khi hai đoạn cách nhau xa (biển chia 2 cột).
4. Chia dòng thành 3 mức theo cỡ chữ, so với chữ lớn nhất trong ảnh:
   - Chính   (>= 75%): luôn đọc; chữ mờ thì báo "Có biển hiệu nhưng chữ chưa rõ."
   - Phụ     (25-75%): đọc nếu độ tin cậy >= 50%.
   - Chi tiết (< 25%): bỏ qua, chỉ đọc khi người dùng yêu cầu (detail=True).
5. Gom dòng liền kề cùng mức thành khối; thứ tự đọc: theo mức (Chính → Phụ → Chi tiết),
   trong mức đọc trên → dưới, các khối ngang hàng đọc khối bên trái trước.
6. Mỗi loại nội dung có cách đọc riêng:
   - Biển hiệu: theo bảng mức trên, giới hạn ~20 từ  → "Biển hiệu: Bún bò Huế. Bún đậu mẹt."
   - Nhãn đồ vật: tên sản phẩm (chữ to nhất) rồi các dòng HSD, NSX, dung tích → "Nhãn: ..."
   - Văn bản/giấy tờ: thứ tự tự nhiên, không lọc theo cỡ chữ → "Văn bản: ..."
"""
import asyncio
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import httpx
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.services.gemini_ocr_service import GeminiOCRError, gemini_ocr_enabled, gemini_read_fragments

# Từ viết tắt phổ biến cần giữ nguyên IN HOA để TTS đánh vần (ATM, WC, ...)
KNOWN_ACRONYMS = {
    "ATM", "WC", "VIP", "TP", "HCM", "HN", "ĐT", "SĐT", "DT", "QL", "KCN", "UBND",
    "BV", "THPT", "THCS", "ĐH", "CĐ", "VN", "USA", "WIFI", "OK", "SOS", "CCCD", "CMND",
    "GPS", "USB", "TV", "PC", "KM", "CM", "MM", "KG", "ML", "HSD", "NSX",
}

# Ký tự rác hay xuất hiện ở viền chữ khi OCR
_EDGE_JUNK = " \t|_~`'\"^*.,;:-–—=+<>\\/[](){}"

# Mức chữ theo cỡ (chiều cao so với chữ lớn nhất trong ảnh)
TIER_MAIN = "main"      # Chính
TIER_SUB = "sub"        # Phụ
TIER_DETAIL = "detail"  # Chi tiết
TIER_RANK = {TIER_MAIN: 0, TIER_SUB: 1, TIER_DETAIL: 2}

# Hai đoạn chữ cùng hàng cách nhau hơn N lần chiều cao chữ → hai cột khác nhau
COLUMN_GAP_FACTOR = 2.0
# Đoạn chữ ngắn nằm trong 1% bề ngang sát mép ảnh → chữ bị cắt của biển khác
EDGE_MARGIN_RATIO = 0.01
# Hai đoạn chữ chỉ được xếp cùng hàng khi đoạn cao hơn không cao quá N lần đoạn thấp hơn
ROW_MAX_HEIGHT_RATIO = 1.8

# Dòng thông tin quan trọng trên nhãn đồ vật
LABEL_KEYWORDS = re.compile(
    r"\b(hsd|nsx|hạn sử dụng|hạn dùng|ngày sản xuất|ngày sx|exp|mfg|best before|"
    r"dung tích|thể tích|khối lượng|trọng lượng|net\s*wt)\b",
    re.IGNORECASE,
)
LABEL_UNIT = re.compile(r"\d+(?:[.,]\d+)?\s?(ml|l|lít|g|gr|gam|kg|mg|oz)\b", re.IGNORECASE)

MSG_NO_TEXT = "Không tìm thấy chữ trong khung hình."
MSG_BLURRY = "Chữ trong ảnh chưa rõ. Hãy giữ máy ổn định và đưa camera lại gần hơn."
MSG_SIGN_BLURRY = "Có biển hiệu nhưng chữ chưa rõ."
MSG_LABEL_BLURRY = "Có nhãn nhưng chữ chưa rõ."

CONTEXT_PREFIX = {
    "sign": "Biển hiệu",
    "label": "Nhãn",
    "text": "Văn bản",
}


@dataclass
class OCRFragment:
    """Một đoạn chữ do EasyOCR trả về (đã quy đổi về tọa độ ảnh gốc)."""
    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    tier_hint: Optional[str] = None  # mức cỡ chữ do engine xếp sẵn (Gemini); None = tính theo chiều cao

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


@dataclass
class OCRLine:
    """Một dòng chữ (các đoạn cùng hàng, cùng cột)."""
    fragments: List[OCRFragment] = field(default_factory=list)
    tier: str = TIER_MAIN
    spoken: bool = False
    reason: str = ""  # lý do đọc / bỏ qua (hiện trên trang /test để dễ kiểm tra)

    @property
    def text(self) -> str:
        return " ".join(f.text for f in sorted(self.fragments, key=lambda f: f.x1))

    def readable_text(self, min_conf: float) -> str:
        """Chỉ giữ các đoạn đủ tin cậy để không đọc chữ sai."""
        return " ".join(
            f.text for f in sorted(self.fragments, key=lambda f: f.x1)
            if f.confidence >= _fragment_threshold(f, min_conf)
        )

    @property
    def confidence(self) -> float:
        total_len = sum(len(f.text) for f in self.fragments) or 1
        return sum(f.confidence * len(f.text) for f in self.fragments) / total_len

    @property
    def bbox(self) -> List[int]:
        return [
            min(f.x1 for f in self.fragments),
            min(f.y1 for f in self.fragments),
            max(f.x2 for f in self.fragments),
            max(f.y2 for f in self.fragments),
        ]

    @property
    def height(self) -> float:
        return float(np.median([f.height for f in self.fragments]))

    @property
    def cy(self) -> float:
        return float(np.mean([f.cy for f in self.fragments]))


Block = List[OCRLine]


def _is_word(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def count_words(text: str) -> int:
    """Đếm từ để giới hạn độ dài câu đọc; ký hiệu đứng riêng ('-', '/') không tính là từ."""
    return sum(_is_word(tok) for tok in text.split())


@dataclass
class SpokenItem:
    """
    Một câu đọc, gồm các phần (dấu nối, chữ, các dòng tạo nên phần đó).
    Giữ từng phần để khi vượt giới hạn từ có thể đọc các dòng đầu còn vừa thay vì bỏ cả câu.
    """
    parts: List[Tuple[str, str, List[OCRLine]]]

    @property
    def text(self) -> str:
        joined = "".join(sep + txt for sep, txt, _ in self.parts)
        return _capitalize_first(re.sub(r"\s+", " ", joined).strip(_EDGE_JUNK))

    @property
    def lines(self) -> List[OCRLine]:
        return [ln for _, _, src in self.parts for ln in src]

    @property
    def words(self) -> int:
        return sum(count_words(txt) for _, txt, _ in self.parts)

    def prefix(self, max_words: int) -> Optional["SpokenItem"]:
        """Các phần đầu (nguyên dòng) vừa trong max_words từ."""
        used, keep = 0, []
        for part in self.parts:
            n = count_words(part[1])
            if used + n > max_words:
                break
            keep.append(part)
            used += n
        return SpokenItem(keep) if keep else None

    def truncated(self, max_words: int) -> "SpokenItem":
        kept, n = [], 0
        for tok in self.text.split():
            n += _is_word(tok)
            if n > max_words:
                break
            kept.append(tok)
        return SpokenItem([("", " ".join(kept).rstrip(_EDGE_JUNK), self.lines)])


# Lý do đọc / bỏ qua từng dòng
REASON_READ = "Đọc"
REASON_DETAIL = "Mức Chi tiết: chỉ đọc khi bật 'Đọc cả chữ nhỏ'"
REASON_BLURRY = "Chữ mờ / độ tin cậy thấp: không đọc để tránh đọc sai"
REASON_SUB_LOW = "Mức Phụ nhưng độ tin cậy dưới ngưỡng"
REASON_WORDS = "Vượt giới hạn số từ của câu đọc"
REASON_NEIGHBOUR = "Nằm chéo góc ngoài vùng chữ Chính (coi là biển bên cạnh)"
REASON_LABEL = "Nhãn: không phải tên sản phẩm / HSD / NSX / dung tích"
REASON_BLURRY_MAIN = "Chữ Chính mờ → báo 'chưa rõ'"


@dataclass
class OCRResult:
    """Kết quả OCR hoàn chỉnh trả cho tầng API."""
    lines: List[OCRLine]
    raw_fragment_count: int
    low_confidence_count: int
    text: str
    description: str
    context: str
    inference_time_ms: float
    has_text: bool = False
    detail: bool = False
    engine: str = "easyocr"
    engine_error: Optional[str] = None  # lý do Gemini lỗi phải quay về EasyOCR


def _fragment_threshold(frag: OCRFragment, min_conf: float) -> float:
    # Ký tự đơn lẻ thường là nhiễu -> yêu cầu tin cậy cao hơn
    return min_conf if len(frag.text) > 1 else max(0.6, min_conf)


def _fix_digit_confusions(token: str) -> str:
    """Sửa lỗi OCR hay gặp trong cụm có số: chữ O/o kẹp cạnh chữ số -> số 0 ('18Oml' -> '180ml')."""
    if not any(ch.isdigit() for ch in token):
        return token
    return re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", token)


def _normalize_fragment_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("_", " ")  # EasyOCR đôi khi trả '_' thay cho dấu cách ('IN_MÀU')
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(_fix_digit_confusions(tok) for tok in text.split(" "))
    return text.strip(_EDGE_JUNK)


def _has_meaningful_chars(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _is_upper_text(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 2:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return upper / len(letters) >= 0.8


def _keep_upper_token(token: str) -> bool:
    """Giữ IN HOA cho từ viết tắt hoặc mã có chứa số (ví dụ 'ATM', 'A12')."""
    core = token.strip(_EDGE_JUNK)
    if not core:
        return False
    if core.upper() in KNOWN_ACRONYMS:
        return True
    # Từ viết tắt không có nguyên âm: 'GPP', 'GPKD', 'TNHH'
    if core.isascii() and core.isalpha() and len(core) <= 5 and not any(ch in "AEIOUYaeiouy" for ch in core):
        return True
    return any(ch.isdigit() for ch in core) and any(ch.isalpha() for ch in core)


def to_readable_case(text: str) -> str:
    """
    Chuyển chữ IN HOA của biển hiệu sang dạng câu thường để TTS đọc tự nhiên
    (tránh đánh vần từng chữ cái), vẫn giữ các từ viết tắt.
    'NHÀ THUỐC PHARMACITY' -> 'Nhà Thuốc Pharmacity'
    """
    if not _is_upper_text(text):
        return text
    return " ".join(_readable_token(tok) for tok in text.split(" "))


def _readable_token(tok: str) -> str:
    """Viết hoa chữ cái đầu mỗi từ (chữ biển hiệu phần lớn là tên riêng: 'AN BÌNH' -> 'An Bình')."""
    if "/" in tok:
        # 'ZALO/ĐT:' -> 'Zalo/ĐT:' (xử lý từng phần để giữ từ viết tắt)
        return "/".join(_readable_token(part) for part in tok.split("/"))
    if _keep_upper_token(tok):
        return tok
    return tok.capitalize()


def _capitalize_first(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


def _truncate_words(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(_EDGE_JUNK)


def is_label_info(text: str) -> bool:
    """Dòng có HSD, NSX, dung tích... hoặc dòng ngắn chứa đơn vị (500ml, 1,5 kg)."""
    return bool(LABEL_KEYWORDS.search(text)) or (len(text.split()) <= 5 and bool(LABEL_UNIT.search(text)))


# ---------------------------------------------------------------------------
# Bố cục: đoạn → dòng → mức → khối → thứ tự đọc
# ---------------------------------------------------------------------------

def _same_row(row: OCRLine, frag: OCRFragment) -> bool:
    """
    Cùng hàng khi cỡ chữ tương đương (cao gấp tối đa ROW_MAX_HEIGHT_RATIO lần) và chồng nhau theo chiều dọc
    >= 50% chiều cao đoạn thấp hơn. Tránh chữ rất to ('AN BÌNH') nuốt dòng chữ nhỏ sát bên ('NHÀ THUỐC').
    """
    # So với dải dọc đại diện của cả hàng (tâm + chiều cao trung vị), không so từng đoạn,
    # để các hàng sát nhau không bị nối dây chuyền thành một hàng.
    row_h = row.height
    lo, hi = sorted((row_h, float(frag.height)))
    if hi / lo > ROW_MAX_HEIGHT_RATIO:
        return False
    overlap = min(row.cy + row_h / 2, frag.y2) - max(row.cy - row_h / 2, frag.y1)
    return overlap >= 0.5 * lo


def group_fragments_into_lines(fragments: List[OCRFragment]) -> List[OCRLine]:
    """
    Gom các đoạn chữ có tâm dọc gần nhau thành một hàng, rồi tách hàng tại khoảng trống lớn
    (biển chia 2 cột) để mỗi cột là một dòng riêng. Sắp xếp trên → dưới, trái → phải.
    """
    if not fragments:
        return []
    rows: List[OCRLine] = []
    for frag in sorted(fragments, key=lambda f: f.cy):
        for row in rows:
            if _same_row(row, frag):
                row.fragments.append(frag)
                break
        else:
            rows.append(OCRLine(fragments=[frag]))

    lines: List[OCRLine] = []
    for row in rows:
        frags = sorted(row.fragments, key=lambda f: f.x1)
        current = [frags[0]]
        for frag in frags[1:]:
            prev = current[-1]
            if frag.x1 - prev.x2 > COLUMN_GAP_FACTOR * max(prev.height, frag.height):
                lines.append(OCRLine(fragments=current))
                current = [frag]
            else:
                current.append(frag)
        lines.append(OCRLine(fragments=current))
    lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return lines


def assign_tiers(lines: List[OCRLine]) -> None:
    """Gán mức Chính / Phụ / Chi tiết theo chiều cao chữ so với dòng chữ lớn nhất."""
    if not lines:
        return
    # Bỏ dòng chỉ có 1 ký tự khi tìm chữ lớn nhất (dễ là nhiễu)
    candidates = [ln for ln in lines if sum(ch.isalnum() for ch in ln.text) >= 2] or lines
    ref = max(ln.height for ln in candidates)
    for ln in lines:
        hints = [f.tier_hint for f in ln.fragments if f.tier_hint in TIER_RANK]
        if hints and len(hints) == len(ln.fragments):
            # Engine đã xếp mức (Gemini nhìn cỡ chữ thật) → dùng mức cao nhất trong dòng
            ln.tier = min(hints, key=TIER_RANK.get)
            continue
        ratio = ln.height / ref
        if ratio >= settings.OCR_TIER_MAIN_RATIO:
            ln.tier = TIER_MAIN
        elif ratio >= settings.OCR_TIER_SUB_RATIO:
            ln.tier = TIER_SUB
        else:
            ln.tier = TIER_DETAIL


def _x_overlap(a: List[int], b: List[int]) -> int:
    return min(a[2], b[2]) - max(a[0], b[0])


def group_lines_into_blocks(lines: List[OCRLine]) -> List[Block]:
    """
    Gom các dòng cùng mức, cỡ chữ tương đương, nằm ngay dưới nhau và chồng nhau theo chiều ngang
    thành một khối (ví dụ biển 2 dòng "NHÀ THUỐC" / "PHARMACITY", hoặc một cột của biển 2 cột).
    """
    blocks: List[Block] = []
    for line in sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0])):
        lb = line.bbox
        for block in blocks:
            prev = block[-1]
            pb = prev.bbox
            h = max(prev.height, line.height)
            if (
                prev.tier == line.tier
                and 0.65 <= line.height / max(prev.height, 1.0) <= 1.55
                and -0.3 * h <= lb[1] - pb[3] <= 1.2 * h
                and _x_overlap(pb, lb) > 0
            ):
                block.append(line)
                break
        else:
            blocks.append([line])
    return blocks


def _block_bbox(block: Block) -> List[int]:
    boxes = [ln.bbox for ln in block]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def order_blocks(blocks: List[Block]) -> List[Block]:
    """Đọc trên → dưới; các khối đứng ngang hàng (chồng nhau >= 50% chiều cao) đọc khối trái trước."""
    rows: List[List[Block]] = []
    for block in sorted(blocks, key=lambda b: (_block_bbox(b)[1], _block_bbox(b)[0])):
        bb = _block_bbox(block)
        if rows:
            ref = _block_bbox(rows[-1][0])
            overlap = min(ref[3], bb[3]) - max(ref[1], bb[1])
            if overlap >= 0.5 * min(ref[3] - ref[1], bb[3] - bb[1]):
                rows[-1].append(block)
                continue
        rows.append([block])
    return [b for row in rows for b in sorted(row, key=lambda b: _block_bbox(b)[0])]


# ---------------------------------------------------------------------------
# Tạo câu đọc
# ---------------------------------------------------------------------------

def block_sentences(block: Block, min_conf: float, low_reason: str = REASON_BLURRY) -> List[SpokenItem]:
    """
    Tạo câu từ một khối (các dòng liền nhau cùng mức). Dòng IN HOA nối tiếp hoặc dòng sau viết thường
    (câu bị xuống dòng) được nối vào câu trước; dòng sau bắt đầu bằng chữ hoa thường → câu mới.
    Dòng không có chữ đủ tin cậy (>= min_conf) bị bỏ với lý do low_reason.
    """
    items: List[SpokenItem] = []
    prev_upper = False
    for ln in block:
        original = ln.readable_text(min_conf)
        readable = re.sub(r"\s+", " ", to_readable_case(original)).strip(_EDGE_JUNK)
        if not readable:
            ln.reason = low_reason
            continue
        is_upper = _is_upper_text(original)
        first_alpha = next((ch for ch in readable if ch.isalpha()), "")
        upper_run = bool(items) and prev_upper and is_upper
        continues = upper_run or (bool(items) and bool(first_alpha) and first_alpha.islower())
        if continues and ":" not in readable and ":" not in items[-1].parts[-1][1]:
            # Dòng IN HOA nối tiếp: dấu phẩy để giọng đọc ngắt nghỉ ("In - Photo, Chụp Hình Thẻ")
            # Chữ viết dọc (mỗi hàng một từ: "TƯ / VẤN / HIỆU") nối bằng dấu cách thành cụm "Tư Vấn Hiệu"
            vertical = count_words(items[-1].parts[-1][1]) == 1 and count_words(readable) == 1
            sep = ", " if upper_run and not vertical else " "
            items[-1].parts.append((sep, readable, [ln]))
        else:
            items.append(SpokenItem([("", readable, [ln])]))
        prev_upper = is_upper
    return items


def take_words(items: List[SpokenItem], max_words: Optional[int], required: int) -> List[SpokenItem]:
    """
    Lấy câu theo thứ tự cho tới giới hạn từ:
    - `required` câu đầu (mức Chính) luôn được đọc, quá dài thì cắt bớt từ.
    - Câu sau không vừa: vẫn đọc các dòng đầu của câu còn vừa; dòng còn lại bỏ (lý do: vượt giới hạn),
      và tiếp tục xét các câu sau (câu ngắn phía sau vẫn có thể được đọc).
    """
    if max_words is None:
        return list(items)
    chosen: List[SpokenItem] = []
    used = 0
    for idx, item in enumerate(items):
        if used + item.words <= max_words:
            chosen.append(item)
            used += item.words
            continue
        if idx < required and used < max_words:
            chosen.append(item.truncated(max_words - used))
            used = max_words
            continue
        part = item.prefix(max_words - used) if idx >= required else None
        if part:
            chosen.append(part)
            used += part.words
        kept = {id(ln) for ln in (part.lines if part else [])}
        for ln in item.lines:
            if id(ln) not in kept:
                ln.reason = REASON_WORDS
    return chosen


def _finish(context: str, items: List[SpokenItem]) -> str:
    for item in items:
        for ln in item.lines:
            ln.spoken = True
            ln.reason = REASON_READ
    body = ". ".join(item.text for item in items if item.text)
    body = _truncate_words(body, settings.OCR_MAX_SPOKEN_CHARS).rstrip(_EDGE_JUNK)
    return f"{CONTEXT_PREFIX[context]}: {body}."


def _overlap_ratio(a0: int, a1: int, b0: int, b1: int) -> float:
    """Tỉ lệ đoạn [a0, a1] nằm trong [b0, b1]."""
    return max(0, min(a1, b1) - max(a0, b0)) / max(1, a1 - a0)


def _is_neighbour_sign(block: Block, main_box: List[int]) -> bool:
    """
    Chỉ dùng khi đọc bằng EasyOCR (Gemini đã được dặn bỏ chữ biển khác trong prompt).
    Khối lệch hẳn khỏi vùng chữ Chính theo CẢ chiều ngang lẫn chiều dọc (nằm chéo góc) → biển bên cạnh.
    Chữ phụ nằm cùng hàng (trái/phải) hoặc cùng cột (trên/dưới) với chữ Chính vẫn được đọc.
    """
    bb = _block_bbox(block)
    horizontal = _overlap_ratio(bb[0], bb[2], main_box[0], main_box[2])
    vertical = _overlap_ratio(bb[1], bb[3], main_box[1], main_box[3])
    return horizontal < 0.5 and vertical < 0.5


def compose_sign(blocks: List[Block], detail: bool = False) -> str:
    """
    Biển hiệu (đúng kế hoạch):
    1. Mức Chính: luôn đọc; nếu không còn chữ Chính nào đủ rõ → "Có biển hiệu nhưng chữ chưa rõ."
    2. Mức Phụ: đọc nếu độ tin cậy >= OCR_SUB_MIN_CONFIDENCE.
    3. Mức Chi tiết: chỉ đọc khi detail=True.
    Thứ tự: Chính → Phụ (→ Chi tiết); trong mỗi mức trên → dưới, khối ngang hàng đọc khối trái trước.
    Giới hạn OCR_SIGN_MAX_WORDS từ (không giới hạn khi detail=True).
    """
    ordered = sorted(order_blocks(blocks), key=lambda b: TIER_RANK[b[0].tier])
    main_lines = [ln for b in ordered if b[0].tier == TIER_MAIN for ln in b]
    use_neighbour_filter = bool(main_lines) and not any(
        f.tier_hint for ln in main_lines for f in ln.fragments
    )
    main_box = _block_bbox(main_lines) if main_lines else None

    main_items: List[SpokenItem] = []
    other_items: List[SpokenItem] = []
    for block in ordered:
        tier = block[0].tier
        if tier == TIER_DETAIL and not detail:
            for ln in block:
                ln.reason = REASON_DETAIL
            continue
        if tier != TIER_MAIN and use_neighbour_filter and _is_neighbour_sign(block, main_box):
            for ln in block:
                ln.reason = REASON_NEIGHBOUR
            continue
        if tier == TIER_MAIN:
            main_items.extend(block_sentences(block, settings.OCR_MIN_CONFIDENCE, REASON_BLURRY_MAIN))
        else:
            other_items.extend(block_sentences(block, settings.OCR_SUB_MIN_CONFIDENCE, REASON_SUB_LOW))

    if not main_items:
        # Chữ chính mờ: báo chưa rõ, không đọc chữ sai
        return MSG_SIGN_BLURRY
    max_words = None if detail else settings.OCR_SIGN_MAX_WORDS
    return _finish("sign", take_words(main_items + other_items, max_words, len(main_items)))


def compose_label(blocks: List[Block], detail: bool = False) -> str:
    """Nhãn đồ vật: tên sản phẩm (chữ Chính) trước, sau đó các dòng HSD, NSX, dung tích, khối lượng."""
    ordered = order_blocks(blocks)
    min_conf = settings.OCR_MIN_CONFIDENCE

    name_items: List[SpokenItem] = []
    for block in ordered:
        if block[0].tier == TIER_MAIN:
            name_items.extend(block_sentences(block, min_conf))

    other_lines = [ln for b in ordered if b[0].tier != TIER_MAIN for ln in b]
    info_items: List[SpokenItem] = []
    i = 0
    while i < len(other_lines):
        ln = other_lines[i]
        text = ln.readable_text(min_conf)
        if not text:
            ln.reason = REASON_BLURRY
        elif detail or is_label_info(text):
            src = [ln]
            # "HSD:" / "Dung tích" đứng riêng → giá trị nằm ở dòng kế tiếp
            value_on_next_line = not any(ch.isdigit() for ch in text) and len(text.split()) <= 4
            if not detail and value_on_next_line and i + 1 < len(other_lines):
                nxt_text = other_lines[i + 1].readable_text(min_conf)
                if nxt_text:
                    text = f"{text} {nxt_text}"
                    src.append(other_lines[i + 1])
                    i += 1
            info_items.append(SpokenItem([("", to_readable_case(text).strip(_EDGE_JUNK), src)]))
        else:
            ln.reason = REASON_LABEL
        i += 1

    if not name_items and not info_items:
        return MSG_LABEL_BLURRY
    if not name_items:
        name_items = [SpokenItem([("", "Tên sản phẩm chưa rõ", [])])]
    max_words = None if detail else settings.OCR_LABEL_MAX_WORDS
    return _finish("label", take_words(name_items + info_items, max_words, len(name_items)))


def compose_text(blocks: List[Block], _detail: bool = False) -> str:
    """Văn bản / giấy tờ: thứ tự tự nhiên trên → dưới, trái → phải, không lọc theo cỡ chữ."""
    del _detail  # cùng chữ ký với compose_sign / compose_label
    items: List[SpokenItem] = []
    for block in order_blocks(blocks):
        items.extend(block_sentences(block, settings.OCR_MIN_CONFIDENCE))
    if not items:
        return MSG_BLURRY
    return _finish("text", items)


COMPOSERS: dict = {"sign": compose_sign, "label": compose_label, "text": compose_text}


def decide_context(lines: List[OCRLine], requested: str = "auto") -> str:
    """
    Chọn loại nội dung:
    - 'label': có dòng HSD / NSX / dung tích / đơn vị (500ml...)
    - 'text' : nhiều dòng (>= 6) và phần lớn cùng cỡ chữ lớn nhất → giấy tờ
    - 'sign' : còn lại
    """
    requested = (requested or "auto").lower()
    if requested in COMPOSERS:
        return requested
    texts = [ln.readable_text(settings.OCR_MIN_CONFIDENCE) for ln in lines]
    if any(t and is_label_info(t) for t in texts):
        return "label"
    readable = [ln for ln, t in zip(lines, texts) if t]
    if len(readable) >= 6 and sum(ln.tier == TIER_MAIN for ln in readable) / len(readable) >= 0.5:
        return "text"
    return "sign"


def drop_edge_fragments(fragments: List[OCRFragment], img_width: Optional[int]) -> List[OCRFragment]:
    """
    Bỏ đoạn chữ chạm mép trái/phải ảnh: chữ bị cắt, thường của biển bên cạnh (chữ 'N' ở mép bị gộp
    thành 'N LẤY LIỀN', 'TRƯỜNG TH' của biển phía sau). Đọc chữ bị cắt dở dễ gây hiểu sai.
    Nếu mọi đoạn đều chạm mép (chụp quá sát) thì giữ nguyên để vẫn đọc được.
    """
    if not img_width:
        return fragments
    edge = max(2, int(EDGE_MARGIN_RATIO * img_width))
    inner = [f for f in fragments if f.x1 > edge and f.x2 < img_width - edge]
    return inner or fragments


def analyze_fragments(
    fragments: List[OCRFragment],
    context: str = "auto",
    detail: bool = False,
    img_width: Optional[int] = None,
) -> Tuple[List[OCRLine], str, str, str]:
    """
    Xử lý bố cục thuần (không cần EasyOCR, dễ kiểm thử).
    Returns: (các dòng theo thứ tự đọc tự nhiên, loại nội dung, văn bản thô, câu đọc)
    """
    lines = group_fragments_into_lines(drop_edge_fragments(fragments, img_width))
    if not lines:
        return [], "text", "", MSG_NO_TEXT

    assign_tiers(lines)
    blocks = group_lines_into_blocks(lines)
    ordered_lines = [ln for b in order_blocks(blocks) for ln in b]
    raw_text = "\n".join(
        t for t in (ln.readable_text(settings.OCR_MIN_CONFIDENCE) for ln in ordered_lines) if t
    )

    requested = (context or "auto").lower()
    if not raw_text and requested not in COMPOSERS:
        for ln in ordered_lines:
            ln.reason = REASON_BLURRY
        return ordered_lines, "text", "", MSG_BLURRY

    ctx = decide_context(lines, requested)
    description = COMPOSERS[ctx](blocks, detail)
    for ln in ordered_lines:
        if not ln.reason:
            ln.reason = "Không đọc"
    return ordered_lines, ctx, raw_text, description


class OCRService:
    """EasyOCR wrapper: nạp model một lần lúc khởi động, suy luận trong thread pool."""

    def __init__(self):
        self.reader = None
        self.languages: List[str] = list(settings.OCR_LANGUAGES)
        self.gpu: bool = False
        self._lock: Optional[asyncio.Lock] = None
        self._is_loaded: bool = False
        self.load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.reader is not None

    def _get_lock(self) -> asyncio.Lock:
        # EasyOCR không an toàn khi gọi song song trên cùng reader -> tuần tự hóa
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def load_model(self) -> None:
        if self.is_loaded:
            return
        import easyocr  # import muộn để server vẫn chạy được chức năng khác nếu thiếu thư viện
        try:
            import torch
            self.gpu = settings.DEVICE.lower() == "cuda" and torch.cuda.is_available()
        except Exception:
            self.gpu = False

        logger.info(f"Loading EasyOCR reader (languages={self.languages}, gpu={self.gpu})...")
        start_t = time.perf_counter()
        try:
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            # Warm-up để request đầu tiên từ ESP32 không bị chậm
            self.reader.readtext(np.full((64, 200, 3), 255, dtype=np.uint8))
            self._is_loaded = True
            self.load_error = None
            logger.info(f"EasyOCR loaded in {(time.perf_counter() - start_t) * 1000:.1f} ms.")
        except Exception as e:
            self._is_loaded = False
            self.load_error = str(e)
            logger.error(f"Failed to load EasyOCR: {e}", exc_info=True)
            raise RuntimeError(f"Could not load EasyOCR: {e}")

    @staticmethod
    def preprocess(img_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Chuẩn hóa kích thước và tăng tương phản cho ảnh ESP32-CAM.
        Returns: (ảnh đã xử lý, hệ số scale so với ảnh gốc)
        """
        h, w = img_rgb.shape[:2]
        long_side = max(h, w)
        # Đưa cạnh dài về OCR_MAX_IMAGE_SIDE (phóng to tối đa 2 lần): chữ nhỏ của ảnh ESP32 rõ hơn
        scale = min(2.0, settings.OCR_MAX_IMAGE_SIDE / long_side)
        if abs(scale - 1.0) > 1e-3:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=interp)

        if not settings.OCR_CLAHE:
            return img_rgb, scale

        # CLAHE trên kênh độ sáng: chỉ bật cho ảnh thiếu sáng / ngược sáng (ảnh biển hiệu ban ngày đọc kém hơn)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2RGB)
        return enhanced, scale

    def _readtext_sync(self, img_rgb: np.ndarray) -> Tuple[List[OCRFragment], int]:
        """
        Trả về các đoạn có độ tin cậy >= OCR_TIER_MIN_CONFIDENCE. Đoạn mờ (dưới OCR_MIN_CONFIDENCE)
        vẫn được giữ để xác định cỡ chữ lớn nhất và báo "chữ chưa rõ", nhưng không được đọc.
        """
        processed, scale = self.preprocess(img_rgb)
        results = self.reader.readtext(processed, detail=1, paragraph=False)

        fragments: List[OCRFragment] = []
        dropped = 0
        for box, text, conf in results:
            text = _normalize_fragment_text(str(text))
            if not text or not _has_meaningful_chars(text):
                continue
            conf = float(conf)
            floor = settings.OCR_TIER_MIN_CONFIDENCE if len(text) > 1 else 0.6
            if conf < floor:
                dropped += 1
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            fragments.append(OCRFragment(
                text=text,
                confidence=conf,
                x1=int(min(xs) / scale),
                y1=int(min(ys) / scale),
                x2=int(max(xs) / scale),
                y2=int(max(ys) / scale),
            ))
        return fragments, dropped

    async def _read_easyocr(self, img_rgb: np.ndarray) -> Tuple[List[OCRFragment], int]:
        if not self.is_loaded:
            # Thử nạp lại nếu lúc khởi động bị lỗi (ví dụ mất mạng khi tải model)
            await asyncio.to_thread(self.load_model)
        async with self._get_lock():
            return await asyncio.wait_for(
                asyncio.to_thread(self._readtext_sync, img_rgb),
                timeout=settings.INFERENCE_TIMEOUT_SECONDS * 2,
            )

    async def read(
        self,
        img_rgb: np.ndarray,
        context: str = "auto",
        detail: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> OCRResult:
        """Nhận dạng bằng Gemini (nếu bật và có client), lỗi thì quay về EasyOCR."""
        start = time.perf_counter()
        engine = "easyocr"
        engine_error: Optional[str] = None
        fragments: Optional[List[OCRFragment]] = None
        dropped = 0

        if gemini_ocr_enabled() and client is not None:
            try:
                fragments = await gemini_read_fragments(img_rgb, client)
                engine = "gemini"
            except GeminiOCRError as e:
                engine_error = str(e)
                logger.warning(f"Gemini OCR failed, falling back to EasyOCR: {engine_error}")

        if fragments is None:
            fragments, dropped = await self._read_easyocr(img_rgb)
        inf_ms = round((time.perf_counter() - start) * 1000.0, 2)

        lines, ctx, text, description = analyze_fragments(fragments, context, detail, img_rgb.shape[1])
        blurry = sum(1 for f in fragments if f.confidence < _fragment_threshold(f, settings.OCR_MIN_CONFIDENCE))

        return OCRResult(
            lines=lines,
            raw_fragment_count=len(fragments) + dropped,
            low_confidence_count=dropped + blurry,
            text=text,
            description=description,
            context=ctx,
            inference_time_ms=inf_ms,
            has_text=any(ln.spoken for ln in lines),
            detail=detail,
            engine=engine,
            engine_error=engine_error,
        )


TIER_COLORS_BGR = {TIER_MAIN: (0, 200, 0), TIER_SUB: (0, 165, 255), TIER_DETAIL: (160, 160, 160)}
TIER_TAGS = {TIER_MAIN: "CHINH", TIER_SUB: "PHU", TIER_DETAIL: "CT"}


def draw_ocr_annotations(img_rgb: np.ndarray, lines: List[OCRLine]) -> np.ndarray:
    """
    Vẽ khung từng dòng chữ: xanh = Chính, cam = Phụ, xám = Chi tiết; nét đậm = được đọc.
    Nhãn gồm số thứ tự, mức và độ tin cậy (cv2 không vẽ được dấu tiếng Việt).
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for idx, line in enumerate(lines, start=1):
        color = TIER_COLORS_BGR.get(line.tier, (0, 200, 0))
        thickness = 2 if line.spoken else 1
        for frag in line.fragments:
            cv2.rectangle(img_bgr, (frag.x1, frag.y1), (frag.x2, frag.y2), color, thickness)
        x1, y1, _, _ = line.bbox
        tag = f"#{idx} {TIER_TAGS.get(line.tier, '')} {int(line.confidence * 100)}%"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(img_bgr, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(img_bgr, tag, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr


# Global singleton instance
ocr_service = OCRService()
