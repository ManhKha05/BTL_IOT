# BTL IoT - Assistive AI Server for Visually Impaired

Hệ thống AI Server hỗ trợ người khiếm thị qua thiết bị đeo dùng ESP32-CAM, xây dựng trên **FastAPI**, **Ultralytics YOLO-World (YOLOv8)**, **EasyOCR** và **Microsoft Edge-TTS / gTTS**.

| Chức năng | Thao tác trên thiết bị | Endpoint server |
|---|---|---|
| Nhận diện vật thể & định hướng | Nhấn nhả nút một lần (Single Click) | `POST /api/process/detect` |
| Đọc chữ trên biển hiệu / nhãn đồ vật | Nhấn giữ nút 1,5 giây (Long Press) | `POST /api/process/ocr` |

---

## 1. Yêu cầu môi trường

- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+ hoặc macOS.
- **Python**: **3.11** (đã kiểm thử trên 3.11.9).
- **GPU (tùy chọn)**: NVIDIA với driver hỗ trợ CUDA 12.1 (đã kiểm thử GTX 1650 4GB). Không có GPU vẫn chạy được trên CPU, chỉ chậm hơn.
- **Ổ đĩa**: khoảng 6 GB cho môi trường ảo (PyTorch CUDA chiếm phần lớn) + ~200 MB trọng số model.
- **Internet**: cần khi cài đặt, khi chạy lần đầu (tải trọng số YOLO-World, CLIP, EasyOCR) và khi tạo giọng nói (Edge-TTS / gTTS là dịch vụ trực tuyến).

---

## 2. Cài đặt

Tất cả lệnh dưới đây chạy từ **thư mục gốc của repo** (thư mục chứa `README.md`).

### Bước 1: Clone dự án
```bash
git clone <URL_REPOSITORY>
cd BTL_IOT
```

### Bước 2: Tạo và kích hoạt môi trường ảo `.venv`
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
  > Nếu PowerShell chặn script, chạy một lần: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, hoặc dùng Command Prompt với `.venv\Scripts\activate.bat`.
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 3: Cài thư viện
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` đã gồm đủ thư viện cho cả hai chức năng (FastAPI, Ultralytics, CLIP, EasyOCR, Edge-TTS, gTTS...).

> **PyTorch:** file đã trỏ tới bản **CUDA 12.1** (`--extra-index-url https://download.pytorch.org/whl/cu121`).
> Máy không có GPU NVIDIA có thể cài bản CPU nhẹ hơn **trước** khi cài phần còn lại:
> ```bash
> pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
> ```
> rồi chạy lại `pip install -r requirements.txt`.

Kiểm tra nhanh sau khi cài:
```bash
python -c "import torch, ultralytics, easyocr, edge_tts; print('CUDA:', torch.cuda.is_available())"
```

### Bước 4: Tạo file cấu hình `server/.env`
Server đọc file `.env` **trong thư mục `server/`**:
- **Windows PowerShell:** `Copy-Item server\.env.example server\.env`
- **Linux / macOS:** `cp server/.env.example server/.env`

Các tham số hay chỉnh:

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `PORT` | Cổng server | `8000` |
| `DEVICE` | `cpu` hoặc `cuda` (dùng GPU cho YOLO và EasyOCR) | `cpu` |
| `YOLO_MODEL_PATH` | Trọng số **YOLO-World** (bắt buộc loại `*-worldv2.pt`) | `models/yolov8s-worldv2.pt` |
| `OCR_LANGUAGES` | Ngôn ngữ OCR | `vi,en` |
| `OCR_MIN_CONFIDENCE` | Ngưỡng tin cậy tối thiểu của đoạn chữ | `0.30` |
| `OCR_TTS_VOICE` | Giọng Edge-TTS cho chức năng đọc chữ | `vi-VN-HoaiMyNeural` |
| `API_KEY` | Nếu đặt, mọi request phải gửi header `X-API-Key` | rỗng (tắt) |

---

## 3. Trọng số model (tự tải lần đầu)

Không cần tải tay — lần khởi động đầu tiên sẽ tự tải:
- **YOLO-World Small** `yolov8s-worldv2.pt` → `server/models/` (Ultralytics). Có thể tải tay [tại đây](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt) và đặt vào `server/models/`.
- **CLIP ViT-B/32** (bộ mã hóa văn bản của YOLO-World) → thư mục cache của người dùng.
- **EasyOCR** (bộ phát hiện chữ CRAFT + bộ nhận dạng tiếng Việt/Latin) → `~/.EasyOCR/model/`.

Vì vậy lần chạy đầu sẽ mất vài phút; các lần sau khởi động nhanh.

> Lưu ý: chỉ dùng trọng số **YOLO-World** (`*-worldv2.pt`). Trọng số YOLOv8 thường (`yolov8n.pt`, `yolov8s.pt`) không hỗ trợ `set_classes()` và server sẽ không nạp được model.

---

## 4. Khởi chạy server

Kích hoạt `.venv` rồi chọn một cách:

```bash
# Cách 1: từ thư mục gốc
python server/main.py

# Cách 2: từ thư mục server
cd server
python main.py

# Cách 3: Uvicorn kèm hot-reload (chạy trong thư mục server)
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Khi log hiện `Application startup complete`, server sẵn sàng. Địa chỉ LAN của máy được in ở dòng `Server IP` — ESP32-CAM gửi request tới địa chỉ đó.

---

## 5. API

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| `POST` | `/api/process/detect` | **Chức năng nhận diện:** ảnh + phát hiện YOLO → trợ lý AI mô tả cảnh → MP3 |
| `POST` | `/api/process/ocr` | **Chức năng đọc chữ:** ảnh → EasyOCR → câu "Nội dung biển hiệu: ..." → MP3 (Edge-TTS) |
| `POST` | `/api/detect` | Chỉ nhận diện (không TTS) |
| `POST` | `/api/detect/visualize` | Trả ảnh JPEG có bounding box |
| `POST` | `/api/ocr` | Chỉ OCR (không TTS) |
| `POST` | `/api/tts` | Chuyển văn bản bất kỳ thành MP3 |
| `GET` | `/api/audio/{audio_id}` | Tải file MP3 (lưu 30 phút) |
| `GET` | `/health` | Trạng thái server, model YOLO và OCR |

Các endpoint nhận ảnh chấp nhận **JPEG/PNG tối đa 5 MB**, gửi theo một trong hai cách:
- `multipart/form-data`, field `file` (Swagger, trình duyệt, Postman);
- raw body với header `Content-Type: image/jpeg` (cách gọn nhất cho ESP32-CAM).

Ví dụ thử OCR bằng curl:
```bash
curl -X POST http://localhost:8000/api/process/ocr -H "Content-Type: image/jpeg" --data-binary @bien_hieu.jpg
```

Phản hồi rút gọn:
```json
{
  "success": true,
  "text": "NHÀ THUỐC PHARMACITY",
  "description": "Nội dung biển hiệu: Nhà thuốc Pharmacity.",
  "audio_url": "http://192.168.1.10:8000/api/audio/9f2c...",
  "lines": [{"text": "NHÀ THUỐC PHARMACITY", "confidence": 0.91, "bbox": [..]}]
}
```

Luồng trên ESP32-CAM: chụp ảnh → `POST` tới endpoint tương ứng với thao tác nút → đọc `audio_url` trong JSON → tải MP3 và phát qua loa.

### Trợ lý mô tả cảnh cho Single Click

Tạo khóa trong [Google AI Studio](https://aistudio.google.com/app/apikey), điền `GEMINI_API_KEY` trong `server/.env` rồi khởi động lại server. Mặc định `SCENE_AI_MODEL=gemini-3.5-flash-lite` và thời gian chờ là 12 giây (`SCENE_AI_TIMEOUT_SECONDS`). Server gửi ảnh JPEG đã thu nhỏ cùng gợi ý YOLO tới Gemini. Dùng khóa thuộc dự án Free Tier để chạy trong [hạn mức miễn phí của Google](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite).

Khi chạy trên mạng dùng chung, có thể giới hạn thiết bị được gọi API bằng `API_KEY` và header `X-API-Key` trên ESP32-CAM.

Mặc định `SCENE_AI_MEDIA_RESOLUTION=LOW` để giảm lượng dữ liệu ảnh Gemini xử lý. Nếu mô tả bỏ sót chi tiết nhỏ, đặt `SCENE_AI_MEDIA_RESOLUTION=MEDIUM` trong `server/.env` rồi khởi động lại. Thiết lập này chỉ áp dụng cho Gemini.

Prompt của trợ lý nằm ở [scene_assistant_prompt.md](server/app/services/scene_assistant_prompt.md). Trợ lý ưu tiên vật cản và hướng, dùng tối đa hai câu tiếng Việt ngắn, và diễn đạt mức độ không chắc chắn nếu ảnh khó đọc. `spoken_objects` liệt kê các phát hiện YOLO mà trợ lý đã nhắc tới. Nếu thiếu khóa hoặc dịch vụ AI lỗi, endpoint vẫn trả kết quả YOLO kèm `description_error` và thử tạo MP3 thông báo. `/api/detect` dùng mô tả cục bộ để kiểm thử.

---

## 6. Kiểm thử

1. **Swagger UI:** <http://localhost:8000/docs> — thử trực tiếp mọi endpoint.
2. **Trang kiểm thử:** <http://localhost:8000/test> — upload ảnh hoặc chụp webcam, chọn chế độ *Nhận diện vật thể* hoặc *Đọc chữ (OCR)*, xem kết quả và nghe giọng đọc.

---

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `Could not load YOLO model` | Kiểm tra `YOLO_MODEL_PATH` là file `*-worldv2.pt`; kiểm tra Internet cho lần tải đầu. |
| `/health` báo `ocr_loaded: false` | Lần đầu EasyOCR cần Internet để tải model vào `~/.EasyOCR/`. Xem log khởi động. |
| `audio_error` trong phản hồi | Edge-TTS / gTTS cần Internet; kết quả nhận diện/OCR vẫn được trả về. |
| ESP32 không kết nối được | Máy chủ và ESP32 phải cùng mạng Wi-Fi; mở cổng 8000 trên Windows Firewall. |
| Chạy chậm trên GPU 4GB | Đặt `MAX_CONCURRENT_REQUESTS=1` hoặc dùng `DEVICE=cpu` cho ổn định. |
