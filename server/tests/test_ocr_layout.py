"""
Kiểm thử cách đọc chữ của chức năng 3 (3 mức cỡ chữ, thứ tự đọc, 3 loại nội dung).
Dùng các đoạn chữ giả lập nên không cần EasyOCR / ảnh thật.

Chạy (trong thư mục server/):  python -m pytest tests -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ocr_service import (  # noqa: E402
    MSG_BLURRY,
    MSG_NO_TEXT,
    MSG_SIGN_BLURRY,
    TIER_DETAIL,
    TIER_MAIN,
    TIER_SUB,
    OCRFragment,
    _normalize_fragment_text,
    analyze_fragments,
    count_words,
)


def frag(text, x, y, w, h, conf=0.9):
    """Đoạn chữ giả lập, chuẩn hóa giống bước đọc EasyOCR thật."""
    return OCRFragment(text=_normalize_fragment_text(text), confidence=conf, x1=x, y1=y, x2=x + w, y2=y + h)


def sign_fragments(main_conf=0.9, sub_conf=0.8):
    """Biển quán: chữ chính 'BÚN BÒ HUẾ', chữ phụ 'BÚN ĐẬU MẸT', chữ nhỏ SĐT và 'Quảng cáo'."""
    return [
        frag("BÚN BÒ HUẾ", 100, 100, 600, 100, main_conf),
        frag("BÚN ĐẬU MẸT", 180, 230, 400, 55, sub_conf),
        frag("SĐT: 0912 345 678", 200, 320, 300, 20),
        frag("Quảng cáo", 600, 400, 120, 18),
    ]


def test_sign_reads_main_then_sub_and_skips_detail():
    lines, ctx, _, desc = analyze_fragments(sign_fragments())
    assert ctx == "sign"
    assert desc == "Biển hiệu: Bún Bò Huế. Bún Đậu Mẹt."
    tiers = {ln.text: ln.tier for ln in lines}
    assert tiers["BÚN BÒ HUẾ"] == TIER_MAIN
    assert tiers["BÚN ĐẬU MẸT"] == TIER_SUB
    assert tiers["SĐT: 0912 345 678"] == TIER_DETAIL
    assert [ln.text for ln in lines if ln.spoken] == ["BÚN BÒ HUẾ", "BÚN ĐẬU MẸT"]


def test_sign_blurry_main_reports_unclear_instead_of_wrong_text():
    _, _, _, desc = analyze_fragments(sign_fragments(main_conf=0.2))
    assert desc == MSG_SIGN_BLURRY


def test_sign_sub_below_50_percent_is_skipped():
    _, _, _, desc = analyze_fragments(sign_fragments(sub_conf=0.4))
    assert desc == "Biển hiệu: Bún Bò Huế."


def test_sign_detail_mode_reads_small_text():
    _, _, _, desc = analyze_fragments(sign_fragments(), detail=True)
    assert "0912 345 678" in desc and "Quảng cáo" in desc
    assert desc.index("Bún Đậu Mẹt") < desc.index("0912")


def test_two_column_sign_reads_left_block_first():
    fragments = [
        frag("PHỞ BÒ", 50, 100, 250, 60),
        frag("CƠM TẤM", 700, 100, 300, 60),
        frag("PHỞ GÀ", 50, 180, 250, 60),
        frag("CƠM GÀ", 700, 180, 300, 60),
    ]
    _, ctx, _, desc = analyze_fragments(fragments)
    assert ctx == "sign"
    low = desc.lower()
    assert low.index("phở bò") < low.index("phở gà") < low.index("cơm tấm") < low.index("cơm gà")


def test_sign_word_limit():
    long_text = " ".join(["chữ"] * 25)
    _, _, _, desc = analyze_fragments([frag(long_text, 0, 0, 900, 80)], context="sign")
    body = desc.split(": ", 1)[1].rstrip(".")
    assert len(body.split()) == 20


def test_label_reads_name_then_key_info_not_ingredients():
    fragments = [
        frag("SỮA TƯƠI VINAMILK", 100, 50, 500, 80),
        frag("Thành phần: sữa bò tươi 99%, đường 1%", 100, 200, 450, 20),
        frag("HSD: 12/05/2027", 100, 240, 200, 22),
        frag("Dung tích", 100, 280, 120, 22),
        frag("180ml", 100, 310, 80, 22),
    ]
    _, ctx, _, desc = analyze_fragments(fragments)
    assert ctx == "label"
    assert desc == "Nhãn: Sữa Tươi Vinamilk. HSD: 12/05/2027. Dung tích 180ml."
    assert "Thành phần" not in desc


def test_document_reads_everything_in_natural_order():
    fragments = [frag(f"Dòng thứ {i}", 50, 50 + i * 40, 300, 25) for i in range(1, 8)]
    _, ctx, _, desc = analyze_fragments(fragments)
    assert ctx == "text"
    assert desc.startswith("Văn bản: Dòng thứ 1")
    assert desc.index("Dòng thứ 1") < desc.index("Dòng thứ 7")


def test_real_photocopy_sign_ignores_neighbour_signs():
    """Biển 'MINH DUY PHOTOCOPY / IN - PHOTO / CHỤP HÌNH THẺ / LẤY LIỀN' (ảnh 600 px), có biển khác ở mép trái."""
    fragments = [
        frag("MINH DUY PHOTOCOPY", 70, 71, 440, 59, 0.96),
        frag("IN", 83, 141, 98, 68, 0.99),
        frag("PHOTO", 239, 125, 264, 93, 0.98),
        frag("A0", 125, 220, 60, 37, 0.50),
        frag("A3", 219, 225, 56, 31, 0.54),
        frag("A4", 311, 220, 60, 37, 0.98),
        frag("CHỤP HÌNH THẺ", 89, 257, 412, 72, 0.93),
        frag("N", 0, 335, 50, 65, 0.80),          # chữ bị cắt của biển bên cạnh, sát mép trái
        frag("LẤY LIỀN", 166, 333, 256, 81, 0.91),
        frag("IN_MÀU", 92, 429, 157, 46, 0.67),
        frag("SCAN MÀU", 279, 423, 222, 49, 0.95),
        frag("IN", 21, 522, 53, 47, 0.91),        # chữ của biển bên trái, ngoài vùng biển chính
        frag("ZALO/ĐT: 0908 819 323", 89, 652, 423, 47, 0.55),
    ]
    lines, ctx, _, desc = analyze_fragments(fragments, img_width=600)
    assert ctx == "sign"
    assert desc.startswith("Biển hiệu: In Photo, Chụp Hình Thẻ, Lấy Liền. Minh Duy Photocopy.")
    assert "In Màu Scan Màu" in desc and "_" not in desc
    assert " n " not in desc.lower()
    neighbour = [ln for ln in lines if ln.text == "IN" and ln.bbox[0] == 21][0]
    assert not neighbour.spoken
    assert count_words(desc.split(": ", 1)[1]) <= 20


def test_no_text_and_all_blurry_messages():
    assert analyze_fragments([])[3] == MSG_NO_TEXT
    assert analyze_fragments([frag("abc", 0, 0, 100, 40, conf=0.2)])[3] == MSG_BLURRY


def test_horizontal_sign_reads_sub_text_beside_main():
    """Biển ngang: chữ Phụ nằm bên trái / phải chữ Chính (cùng hàng) vẫn là của biển chính."""
    fragments = [
        frag("NHÀ THUỐC", 20, 135, 210, 35),        # bên trái, cùng hàng
        frag("HÀ CHÂU", 260, 110, 250, 80),          # chữ Chính
        frag("BÁN LẺ THUỐC TÂY", 530, 135, 200, 35),  # bên phải, cùng hàng
    ]
    lines, _, _, desc = analyze_fragments(fragments, img_width=780)
    assert desc == "Biển hiệu: Hà Châu. Nhà Thuốc. Bán Lẻ Thuốc Tây."
    assert all(ln.spoken for ln in lines)


def test_short_sub_line_is_not_lost_with_long_neighbour():
    """Dòng Phụ ngắn bị gộp khối với dòng Phụ rất dài: vẫn đọc dòng ngắn, chỉ bỏ dòng vượt giới hạn từ."""
    long_line = "BÁN LẺ THUỐC THÀNH PHẨM DỤNG CỤ Y TẾ VẬT TƯ Y TẾ MỸ PHẨM THỰC PHẨM CHỨC NĂNG"
    fragments = [
        frag("PHONG PHÚ", 100, 50, 360, 60),
        frag("ĐẠT CHUẨN GPP", 200, 130, 150, 25),
        frag(long_line, 70, 160, 420, 24),
    ]
    lines, _, _, desc = analyze_fragments(fragments, img_width=565)
    assert desc == "Biển hiệu: Phong Phú. Đạt Chuẩn GPP."
    reasons = {ln.text: ln.reason for ln in lines}
    assert reasons["ĐẠT CHUẨN GPP"] == "Đọc"
    assert "giới hạn" in reasons[long_line]


def test_every_line_has_a_reason():
    lines, _, _, _ = analyze_fragments(sign_fragments(sub_conf=0.4))
    reasons = {ln.text: ln.reason for ln in lines}
    assert reasons["BÚN BÒ HUẾ"] == "Đọc"
    assert "Phụ" in reasons["BÚN ĐẬU MẸT"]
    assert "Chi tiết" in reasons["SĐT: 0912 345 678"]


def test_symbols_are_not_counted_as_words_and_vertical_text_joins():
    assert count_words("A0 - A3 - A4 - A5") == 4
    fragments = [
        frag("PHỞ", 100, 50, 300, 90),
        frag("TƯ", 150, 300, 40, 40),
        frag("VẤN", 150, 345, 40, 40),
        frag("HIỆU", 150, 390, 40, 40),
    ]
    _, _, _, desc = analyze_fragments(fragments, context="sign")
    assert "Tư Vấn Hiệu" in desc
