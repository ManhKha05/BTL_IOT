import os
import io
import time
import json
import asyncio
import datetime
import socket
from typing import List, Dict, Set
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import easyocr
import edge_tts

app = FastAPI(title="AI Vision Necklace Backend for Visually Impaired")

# Bật CORS để cho phép frontend/web dashboard kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Thư mục lưu trữ tĩnh
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
HISTORY_DIR = os.path.join(UPLOADS_DIR, "history")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Từ điển dịch các vật thể COCO sang tiếng Việt
VIETNAMESE_LABELS: Dict[str, str] = {
    "person": "người",
    "bicycle": "xe đạp",
    "car": "ô tô",
    "motorcycle": "xe máy",
    "airplane": "máy bay",
    "bus": "xe buýt",
    "train": "tàu hỏa",
    "truck": "xe tải",
    "boat": "thuyền",
    "traffic light": "đèn giao thông",
    "fire hydrant": "trụ cứu hỏa",
    "stop sign": "biển báo dừng",
    "bench": "ghế dài",
    "bird": "con chim",
    "cat": "con mèo",
    "dog": "con chó",
    "horse": "con ngựa",
    "backpack": "ba lô",
    "umbrella": "chiếc ô",
    "handbag": "túi xách",
    "bottle": "chai nước",
    "chair": "chiếc ghế",
    "couch": "ghế sô pha",
    "potted plant": "chậu cây",
    "bed": "chiếc giường",
    "dining table": "bàn ăn",
    "toilet": "nhà vệ sinh",
    "tv": "ti vi",
    "laptop": "máy tính xách tay",
    "cell phone": "điện thoại",
    "door": "cửa",
    "stairs": "cầu thang"
}

# Khởi tạo mô hình AI
print("Đang tải mô hình YOLOv8n...")
yolo_model = YOLO("yolov8n.pt")  # Tự động tải yolov8n nếu chưa có

print("Đang khởi tạo EasyOCR (hỗ trợ Tiếng Việt & Tiếng Anh)...")
ocr_reader = easyocr.Reader(['vi', 'en'], gpu=False)

VOICE_NAME = "vi-VN-HoaiMyNeural"  # Giọng nữ tiếng Việt chuẩn Microsoft Edge

def get_local_ip() -> str:
    """Lấy địa chỉ IPv4 nội mạng LAN của máy tính"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Bộ nhớ trạng thái ảnh chụp gần nhất từ ESP32
latest_state = {
    "has_data": False,
    "timestamp": 0,
    "time_str": "Chưa có ảnh nào từ ESP32",
    "mode": "",
    "resolution": "--",
    "size_kb": 0,
    "raw_url": "",
    "annotated_url": "",
    "description": "Đang chờ ảnh chụp đầu tiên từ ESP32-CAM...",
    "extracted_text": "",
    "audio_url": ""
}

# Danh sách lịch sử 12 ảnh gần nhất
capture_history: List[Dict] = []

# Quản lý danh sách kết nối SSE thời gian thực
sse_subscribers: Set[asyncio.Queue] = set()

async def broadcast_update():
    """Gửi cập nhật tức thì tới tất cả trình duyệt đang mở Dashboard"""
    msg = json.dumps(latest_state, ensure_ascii=False)
    for q in list(sse_subscribers):
        try:
            await q.put(msg)
        except Exception:
            sse_subscribers.discard(q)

async def generate_speech(text: str, filename: str) -> str:
    """Tạo file âm thanh từ văn bản sử dụng Edge-TTS"""
    output_path = os.path.join(AUDIO_DIR, filename)
    communicate = edge_tts.Communicate(text, VOICE_NAME)
    await communicate.save(output_path)
    return f"/static/audio/{filename}"

def analyze_spatial_layout(results, img_width: int, img_height: int) -> str:
    """Phân tích vị trí không gian (Trái, Giữa, Phải) và khoảng cách tương đối"""
    detections = []
    frame_area = img_width * img_height

    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if conf < 0.45:
            continue

        raw_name = results[0].names[cls_id]
        vn_name = VIETNAMESE_LABELS.get(raw_name, raw_name)

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_center_x = (x1 + x2) / 2
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area

        # Xác định hướng
        if box_center_x < img_width * 0.35:
            position = "bên trái"
        elif box_center_x > img_width * 0.65:
            position = "bên phải"
        else:
            position = "phía trước"

        # Ước lượng khoảng cách qua tỉ lệ khung hình
        if area_ratio > 0.20:
            distance = "ở rất gần"
        elif area_ratio > 0.07:
            distance = "ở khoảng cách vừa"
        else:
            distance = "ở phía xa"

        detections.append({
            "name": vn_name,
            "position": position,
            "distance": distance,
            "conf": conf,
            "area_ratio": area_ratio
        })

    if not detections:
        return "Không phát hiện thấy vật thể rõ ràng phía trước."

    # Sắp xếp ưu tiên vật thể gần nhất và ở giữa nhất
    detections.sort(key=lambda d: d["area_ratio"], reverse=True)
    top_detections = detections[:3]

    descriptions = []
    for d in top_detections:
        descriptions.append(f"{d['position']} có {d['name']} {d['distance']}")

    summary = ". ".join(descriptions) + "."
    return summary

def draw_ocr_boxes(img_np: np.ndarray, ocr_results: list) -> np.ndarray:
    """Vẽ bounding box và nhãn chữ lên ảnh"""
    annotated = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for bbox, text, prob in ocr_results:
        if prob > 0.30:
            pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            x, y = int(bbox[0][0]), int(bbox[0][1])
            cv2.putText(annotated, text, (x, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return annotated

# ==================== CÁC ROUTE CỦA HỆ THỐNG ====================

@app.get("/")
def serve_dashboard():
    """Trang chủ: Trả về trực tiếp Dashboard Web"""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "AI Vision Necklace Server is running"}

@app.get("/api/server-info")
def get_server_info():
    """Cung cấp IP máy tính và endpoint để cấu hình ESP32 nhanh chóng"""
    local_ip = get_local_ip()
    return {
        "status": "online",
        "local_ip": local_ip,
        "port": 8000,
        "endpoints": {
            "detect": f"http://{local_ip}:8000/api/detect",
            "ocr": f"http://{local_ip}:8000/api/ocr",
            "events": f"http://{local_ip}:8000/api/events",
            "latest": f"http://{local_ip}:8000/api/latest"
        }
    }

@app.get("/api/latest")
def get_latest():
    """Lấy dữ liệu ảnh và kết quả phân tích gần nhất"""
    return {
        **latest_state,
        "history": capture_history
    }

@app.get("/api/events")
async def events(request: Request):
    """Server-Sent Events (SSE) để Dashboard nhận ảnh mới tức thì mà không cần reload"""
    queue = asyncio.Queue()
    sse_subscribers.add(queue)

    async def event_generator():
        try:
            # Gửi dữ liệu hiện tại ngay khi client kết nối
            init_data = json.dumps(latest_state, ensure_ascii=False)
            yield f"data: {init_data}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    # Gửi heartbeat giữ kết nối
                    yield ": heartbeat\n\n"
        finally:
            sse_subscribers.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """API nhận ảnh từ ESP32-CAM, lưu ảnh gốc, nhận diện vật thể và cập nhật giao diện web tức thì"""
    try:
        image_bytes = await file.read()
        ts_now = time.time()
        time_tag = int(ts_now * 1000)
        dt_str = datetime.datetime.now().strftime("%H:%M:%S (%d/%m/%Y)")

        # 1. Lưu ảnh gốc không nén lại từ ESP32
        raw_filename = f"raw_{time_tag}.jpg"
        latest_raw_path = os.path.join(UPLOADS_DIR, "latest_raw.jpg")
        history_raw_path = os.path.join(HISTORY_DIR, raw_filename)
        
        with open(latest_raw_path, "wb") as f:
            f.write(image_bytes)
        with open(history_raw_path, "wb") as f:
            f.write(image_bytes)

        # 2. Xử lý ảnh bằng YOLO
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_image)
        h, w, _ = img_np.shape

        results = yolo_model(img_np)
        description = analyze_spatial_layout(results, w, h)

        # 3. Tạo và lưu ảnh đã vẽ Bounding Box
        annotated_bgr = results[0].plot()
        annotated_filename = f"annotated_{time_tag}.jpg"
        latest_annotated_path = os.path.join(UPLOADS_DIR, "latest_annotated.jpg")
        history_annotated_path = os.path.join(HISTORY_DIR, annotated_filename)

        cv2.imwrite(latest_annotated_path, annotated_bgr)
        cv2.imwrite(history_annotated_path, annotated_bgr)

        # 4. Tạo giọng nói tiếng Việt
        audio_filename = f"detect_{time_tag}.mp3"
        audio_url = await generate_speech(description, audio_filename)

        # 5. Cập nhật state
        latest_state["has_data"] = True
        latest_state["timestamp"] = ts_now
        latest_state["time_str"] = dt_str
        latest_state["mode"] = "detect"
        latest_state["resolution"] = f"{w}x{h}"
        latest_state["size_kb"] = round(len(image_bytes) / 1024, 1)
        latest_state["raw_url"] = f"/static/uploads/latest_raw.jpg?t={time_tag}"
        latest_state["annotated_url"] = f"/static/uploads/latest_annotated.jpg?t={time_tag}"
        latest_state["description"] = description
        latest_state["extracted_text"] = ""
        latest_state["audio_url"] = audio_url

        # Lưu lịch sử (tối đa 12 ảnh)
        capture_history.insert(0, {
            "id": time_tag,
            "time_str": dt_str,
            "mode": "Nhận diện vật thể",
            "raw_url": f"/static/uploads/history/{raw_filename}",
            "annotated_url": f"/static/uploads/history/{annotated_filename}",
            "text": description,
            "audio_url": audio_url
        })
        if len(capture_history) > 12:
            capture_history.pop()

        # Phát sóng SSE tới Dashboard web ngay lập tức
        await broadcast_update()

        return {
            "success": True,
            "type": "detect",
            "description": description,
            "audio_url": audio_url,
            "raw_image_url": latest_state["raw_url"],
            "annotated_image_url": latest_state["annotated_url"],
            "resolution": latest_state["resolution"],
            "size_kb": latest_state["size_kb"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ocr")
async def read_text(file: UploadFile = File(...)):
    """API nhận ảnh từ ESP32-CAM, lưu ảnh gốc, đọc chữ Tiếng Việt và cập nhật giao diện web tức thì"""
    try:
        image_bytes = await file.read()
        ts_now = time.time()
        time_tag = int(ts_now * 1000)
        dt_str = datetime.datetime.now().strftime("%H:%M:%S (%d/%m/%Y)")

        # 1. Lưu ảnh gốc không nén lại từ ESP32
        raw_filename = f"raw_{time_tag}.jpg"
        latest_raw_path = os.path.join(UPLOADS_DIR, "latest_raw.jpg")
        history_raw_path = os.path.join(HISTORY_DIR, raw_filename)
        
        with open(latest_raw_path, "wb") as f:
            f.write(image_bytes)
        with open(history_raw_path, "wb") as f:
            f.write(image_bytes)

        # 2. Xử lý ảnh OCR
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_image)
        h, w, _ = img_np.shape

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        ocr_results = ocr_reader.readtext(gray)
        extracted_texts = [item[1] for item in ocr_results if item[2] > 0.35]

        if extracted_texts:
            full_text = " ".join(extracted_texts)
            speech_text = f"Đọc được nội dung: {full_text}"
        else:
            full_text = ""
            speech_text = "Không tìm thấy văn bản nào trong tầm nhìn."

        # 3. Vẽ khung chữ OCR lên ảnh
        annotated_bgr = draw_ocr_boxes(img_np, ocr_results)
        annotated_filename = f"annotated_{time_tag}.jpg"
        latest_annotated_path = os.path.join(UPLOADS_DIR, "latest_annotated.jpg")
        history_annotated_path = os.path.join(HISTORY_DIR, annotated_filename)

        cv2.imwrite(latest_annotated_path, annotated_bgr)
        cv2.imwrite(history_annotated_path, annotated_bgr)

        # 4. Tạo giọng nói tiếng Việt
        audio_filename = f"ocr_{time_tag}.mp3"
        audio_url = await generate_speech(speech_text, audio_filename)

        # 5. Cập nhật state
        latest_state["has_data"] = True
        latest_state["timestamp"] = ts_now
        latest_state["time_str"] = dt_str
        latest_state["mode"] = "ocr"
        latest_state["resolution"] = f"{w}x{h}"
        latest_state["size_kb"] = round(len(image_bytes) / 1024, 1)
        latest_state["raw_url"] = f"/static/uploads/latest_raw.jpg?t={time_tag}"
        latest_state["annotated_url"] = f"/static/uploads/latest_annotated.jpg?t={time_tag}"
        latest_state["description"] = speech_text
        latest_state["extracted_text"] = full_text
        latest_state["audio_url"] = audio_url

        # Lưu lịch sử (tối đa 12 ảnh)
        capture_history.insert(0, {
            "id": time_tag,
            "time_str": dt_str,
            "mode": "Đọc chữ (OCR)",
            "raw_url": f"/static/uploads/history/{raw_filename}",
            "annotated_url": f"/static/uploads/history/{annotated_filename}",
            "text": speech_text,
            "audio_url": audio_url
        })
        if len(capture_history) > 12:
            capture_history.pop()

        # Phát sóng SSE tới Dashboard web ngay lập tức
        await broadcast_update()

        return {
            "success": True,
            "type": "ocr",
            "extracted_text": full_text,
            "description": speech_text,
            "audio_url": audio_url,
            "raw_image_url": latest_state["raw_url"],
            "annotated_image_url": latest_state["annotated_url"],
            "resolution": latest_state["resolution"],
            "size_kb": latest_state["size_kb"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
