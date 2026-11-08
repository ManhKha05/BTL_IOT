"""
OCR Service for Assistive AI Server (Chức năng 3: đọc chữ trên biển hiệu và nhãn đồ vật).

Pipeline:
1. Tiền xử lý ảnh ESP32-CAM: phóng to ảnh nhỏ, giới hạn ảnh lớn, tăng tương phản (CLAHE).
2. EasyOCR (vi + en) phát hiện và nhận dạng từng đoạn chữ.
3. Lọc nhiễu: bỏ đoạn có độ tin cậy thấp hoặc không chứa chữ/số.
4. Gom các đoạn thành dòng (theo trục dọc), sắp xếp trên → dưới, trái → phải.
5. Gom các dòng liền kề cùng cỡ chữ thành khối (ví dụ biển 2 dòng "NHÀ THUỐC" / "PHARMACITY").
6. Chuẩn hóa chữ (NFC, chữ IN HOA → chữ thường dễ đọc) và tạo câu hoàn chỉnh:
   "Nội dung biển hiệu: Nhà thuốc Pharmacity."
"""
import asyncio
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import logger

# Từ viết tắt phổ biến cần giữ nguyên IN HOA để TTS đánh vần (ATM, WC, ...)
KNOWN_ACRONYMS = {
    "ATM", "WC", "VIP", "TP", "HCM", "HN", "ĐT", "SĐT", "DT", "QL", "KCN", "UBND",
    "BV", "THPT", "THCS", "ĐH", "CĐ", "VN", "USA", "WIFI", "OK", "SOS", "CCCD", "CMND",
    "GPS", "USB", "TV", "PC", "KM", "CM", "MM", "KG", "ML", "HSD", "NSX",
}

# Ký tự rác hay xuất hiện ở viền chữ khi OCR
_EDGE_JUNK = " \t|_~`'\"^*.,;:-–—=+<>\\/[](){}"


@dataclass
class OCRFragment:
    """Một đoạn chữ do EasyOCR trả về (đã quy đổi về tọa độ ảnh gốc)."""
    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


@dataclass
class OCRLine:
    """Một dòng chữ sau khi gom các đoạn nằm cùng hàng."""
    fragments: List[OCRFragment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(f.text for f in sorted(self.fragments, key=lambda f: f.x1))

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


def _fix_digit_confusions(token: str) -> str:
    """Sửa lỗi OCR hay gặp trong cụm có số: chữ O/o kẹp cạnh chữ số -> số 0 ('18Oml' -> '180ml')."""
    if not any(ch.isdigit() for ch in token):
        return token
    return re.sub(r"(?<=\d)[Oo]|[Oo](?=\d)", "0", token)


def _normalize_fragment_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
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
    return any(ch.isdigit() for ch in core) and any(ch.isalpha() for ch in core)


def to_readable_case(text: str) -> str:
    """
    Chuyển chữ IN HOA của biển hiệu sang dạng câu thường để TTS đọc tự nhiên
    (tránh đánh vần từng chữ cái), vẫn giữ các từ viết tắt.
    'NHÀ THUỐC PHARMACITY' -> 'nhà thuốc Pharmacity' (chữ đầu câu được viết hoa ở bước sau)
    """
    if not _is_upper_text(text):
        return text
    out = []
    for tok in text.split(" "):
        if _keep_upper_token(tok):
            out.append(tok)
        elif tok.isascii() and sum(ch.isalpha() for ch in tok) >= 4:
            # Từ không dấu, dài -> thường là tên riêng / thương hiệu: 'PHARMACITY' -> 'Pharmacity'
            out.append(tok.capitalize())
        else:
            out.append(tok.lower())
    return " ".join(out)


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


def group_fragments_into_lines(fragments: List[OCRFragment]) -> List[OCRLine]:
    """Gom các đoạn chữ có tâm dọc gần nhau thành một dòng, sắp xếp trên → dưới."""
    if not fragments:
        return []
    frags = sorted(fragments, key=lambda f: f.cy)
    lines: List[OCRLine] = []
    for frag in frags:
        placed = False
        for line in lines:
            tol = 0.5 * max(line.height, frag.height)
            if abs(line.cy - frag.cy) <= tol:
                line.fragments.append(frag)
                placed = True
                break
        if not placed:
            lines.append(OCRLine(fragments=[frag]))
    lines.sort(key=lambda ln: (ln.bbox[1], ln.bbox[0]))
    return lines


def group_lines_into_blocks(lines: List[OCRLine]) -> List[List[OCRLine]]:
    """
    Gom các dòng liền kề có cỡ chữ tương đương và khoảng cách dọc nhỏ thành một khối nội dung.
    Mỗi khối được đọc thành một câu; các khối khác nhau ngăn cách bằng dấu chấm.
    """
    blocks: List[List[OCRLine]] = []
    for line in lines:
        if blocks:
            prev = blocks[-1][-1]
            ratio = line.height / max(prev.height, 1.0)
            gap = line.bbox[1] - prev.bbox[3]
            if 0.65 <= ratio <= 1.55 and gap <= 1.2 * max(prev.height, line.height):
                blocks[-1].append(line)
                continue
        blocks.append([line])
    return blocks


def decide_context(lines: List[OCRLine], img_height: int, requested: str = "auto") -> str:
    """
    Xác định loại nội dung để chọn câu dẫn:
    - 'sign'  : biển hiệu (ít dòng, chữ lớn) -> "Nội dung biển hiệu: ..."
    - 'label' : nhãn đồ vật                  -> "Nội dung nhãn: ..."
    - 'text'  : không chắc chắn               -> "Nội dung đọc được: ..."
    """
    requested = (requested or "auto").lower()
    if requested in ("sign", "label", "text"):
        return requested
    if not lines:
        return "text"
    # Biển hiệu: ít dòng, chữ lớn, tối đa 2 khối (tên + dòng phụ), không có dạng "Trường: giá trị"
    max_height_ratio = max(ln.height for ln in lines) / max(img_height, 1)
    blocks = group_lines_into_blocks(lines)
    has_field_syntax = any(":" in ln.text for ln in lines)
    if len(lines) <= 3 and len(blocks) <= 2 and max_height_ratio >= 0.06 and not has_field_syntax:
        return "sign"
    return "text"


CONTEXT_PREFIX = {
    "sign": "Nội dung biển hiệu",
    "label": "Nội dung nhãn",
    "text": "Nội dung đọc được",
}


def build_spoken_sentence(lines: List[OCRLine], context: str, max_chars: int) -> Tuple[str, str]:
    """
    Tạo (text_thô, câu_đọc) từ các dòng chữ.
    text_thô giữ nguyên chữ OCR (nối dòng bằng '\\n'), câu_đọc là câu hoàn chỉnh cho TTS.
    """
    raw_text = "\n".join(ln.text for ln in lines)
    blocks = group_lines_into_blocks(lines)
    sentences: List[str] = []
    for block in blocks:
        # Trong một khối: nối dòng bằng dấu cách nếu là chữ IN HOA của biển hiệu hoặc dòng sau
        # viết thường (câu bị xuống dòng); nếu dòng sau bắt đầu bằng chữ hoa -> cụm ý mới.
        parts: List[str] = []
        prev_upper = False
        for ln in block:
            original = ln.text
            readable = to_readable_case(original).strip(_EDGE_JUNK)
            if not readable:
                continue
            is_upper = _is_upper_text(original)
            first_alpha = next((ch for ch in readable if ch.isalpha()), "")
            continues = bool(parts) and ((prev_upper and is_upper) or (first_alpha and first_alpha.islower()))
            if continues and ":" not in readable and ":" not in parts[-1]:
                parts[-1] = f"{parts[-1]} {readable}"
            else:
                parts.append(readable)
            prev_upper = is_upper
        for part in parts:
            part = re.sub(r"\s+", " ", part).strip(_EDGE_JUNK)
            if part:
                sentences.append(_capitalize_first(part))

    body = ". ".join(sentences)
    body = _truncate_words(body, max_chars)
    body = body.rstrip(_EDGE_JUNK)
    prefix = CONTEXT_PREFIX.get(context, CONTEXT_PREFIX["text"])
    return raw_text, f"{prefix}: {body}."


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
        scale = 1.0
        if long_side < 1000:
            scale = min(2.0, 1280.0 / long_side)          # ảnh VGA/SVGA từ ESP32 -> phóng to
        elif long_side > settings.OCR_MAX_IMAGE_SIDE:
            scale = settings.OCR_MAX_IMAGE_SIDE / long_side  # ảnh quá lớn -> thu nhỏ cho nhanh
        if abs(scale - 1.0) > 1e-3:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=interp)

        # CLAHE trên kênh độ sáng giúp chữ nổi hơn với ảnh thiếu sáng / ngược sáng
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l_ch), a_ch, b_ch)), cv2.COLOR_LAB2RGB)
        return enhanced, scale

    def _readtext_sync(self, img_rgb: np.ndarray) -> Tuple[List[OCRFragment], int]:
        processed, scale = self.preprocess(img_rgb)
        results = self.reader.readtext(processed, detail=1, paragraph=False)

        fragments: List[OCRFragment] = []
        low_conf = 0
        for box, text, conf in results:
            text = _normalize_fragment_text(str(text))
            if not text or not _has_meaningful_chars(text):
                continue
            conf = float(conf)
            # Ký tự đơn lẻ thường là nhiễu -> yêu cầu tin cậy cao hơn
            min_conf = settings.OCR_MIN_CONFIDENCE if len(text) > 1 else max(0.6, settings.OCR_MIN_CONFIDENCE)
            if conf < min_conf:
                low_conf += 1
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
        return fragments, low_conf

    async def read(self, img_rgb: np.ndarray, context: str = "auto") -> OCRResult:
        if not self.is_loaded:
            # Thử nạp lại nếu lúc khởi động bị lỗi (ví dụ mất mạng khi tải model)
            await asyncio.to_thread(self.load_model)

        start = time.perf_counter()
        async with self._get_lock():
            fragments, low_conf = await asyncio.wait_for(
                asyncio.to_thread(self._readtext_sync, img_rgb),
                timeout=settings.INFERENCE_TIMEOUT_SECONDS * 2,
            )
        inf_ms = round((time.perf_counter() - start) * 1000.0, 2)

        lines = group_fragments_into_lines(fragments)
        img_h = img_rgb.shape[0]

        if lines:
            ctx = decide_context(lines, img_h, context)
            text, description = build_spoken_sentence(lines, ctx, settings.OCR_MAX_SPOKEN_CHARS)
        elif low_conf > 0:
            ctx = "text"
            text = ""
            description = "Chữ trong ảnh chưa rõ. Hãy giữ máy ổn định và đưa camera lại gần hơn."
        else:
            ctx = "text"
            text = ""
            description = "Không tìm thấy chữ trong khung hình."

        return OCRResult(
            lines=lines,
            raw_fragment_count=len(fragments) + low_conf,
            low_confidence_count=low_conf,
            text=text,
            description=description,
            context=ctx,
            inference_time_ms=inf_ms,
        )


def draw_ocr_annotations(img_rgb: np.ndarray, lines: List[OCRLine]) -> np.ndarray:
    """Vẽ khung từng dòng chữ kèm số thứ tự đọc (cv2 không vẽ được dấu tiếng Việt)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for idx, line in enumerate(lines, start=1):
        for frag in line.fragments:
            cv2.rectangle(img_bgr, (frag.x1, frag.y1), (frag.x2, frag.y2), (0, 200, 0), 2)
        x1, y1, _, _ = line.bbox
        tag = f"#{idx} {int(line.confidence * 100)}%"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(img_bgr, (x1, top), (x1 + tw + 6, top + th + base + 4), (0, 200, 0), -1)
        cv2.putText(img_bgr, tag, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr


# Global singleton instance
ocr_service = OCRService()
