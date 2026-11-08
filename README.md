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
| `POST` | `/api/process/ocr` | **Chức năng đọc chữ:** ảnh → Gemini (dự phòng EasyOCR) → lọc theo cỡ chữ → câu "Biển hiệu: ..." → MP3 (Edge-TTS). Query: `context`, `detail` |
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
  "context": "sign",
  "text": "BÚN BÒ HUẾ\nBÚN ĐẬU MẸT",
  "description": "Biển hiệu: Bún Bò Huế. Bún Đậu Mẹt.",
  "engine": "gemini",
  "audio_url": "http://192.168.1.10:8000/api/audio/9f2c...",
  "lines": [
    {"text": "BÚN BÒ HUẾ", "confidence": 0.91, "bbox": [..], "tier": "main", "spoken": true},
    {"text": "BÚN ĐẬU MẸT", "confidence": 0.84, "bbox": [..], "tier": "sub", "spoken": true},
    {"text": "SĐT: 0912 345 678", "confidence": 0.88, "bbox": [..], "tier": "detail", "spoken": false}
  ]
}
```

### Cách đọc chữ (chức năng 3)

**0. Nhận dạng chữ.** Khi đã điền `GEMINI_API_KEY`, server dùng Gemini (`OCR_GEMINI_MODEL=gemini-3.1-flash-lite`) để đọc chữ. Gemini trả về từng dòng chữ kèm khung, độ rõ và **mức cỡ chữ** do nó nhìn ảnh xếp theo quy tắc ở mục 1. Lý do: khung chữ của Gemini chỉ là ước lượng và xê dịch giữa các lần gọi, nên không dùng để tính tỉ lệ. Đo trên laptop CPU 4 nhân, Gemini mất khoảng 2–5 giây mỗi ảnh, còn EasyOCR mất 12–20 giây. Nếu thiếu khoá, mất mạng, hết hạn mức hoặc quá `OCR_GEMINI_TIMEOUT_SECONDS` (10 giây), server tự chuyển sang EasyOCR. Khi đó mức cỡ chữ được tính theo chiều cao khung. Phản hồi có trường `engine` (`gemini`/`easyocr`) và `engine_error`. Muốn chỉ dùng EasyOCR thì đặt `OCR_ENGINE=easyocr`.

**1. Ba mức cỡ chữ.** Cỡ chữ là chiều cao của chữ, so với dòng chữ lớn nhất trong ảnh:

| Mức | Cỡ chữ | Xử lý |
|---|---|---|
| Chính (`main`) | ≥ 75% | Luôn đọc. Nếu chữ mờ thì nói *"Có biển hiệu nhưng chữ chưa rõ."*, không đọc chữ sai |
| Phụ (`sub`) | 25–75% | Đọc nếu độ tin cậy ≥ 50% |
| Chi tiết (`detail`) | < 25% | Mặc định bỏ qua. Chỉ đọc khi gửi `?detail=true` |

**2. Thứ tự đọc.** Đọc mức Chính trước rồi đến mức Phụ. Trong cùng một mức, đọc từ trên xuống dưới, mỗi dòng đọc từ trái sang phải. Biển chia 2 cột thì đọc hết cột trái trước.

**3. Mỗi loại nội dung có cách đọc riêng.** Server tự nhận loại, hoặc bạn chỉ định bằng `?context=`:
- `sign` (biển hiệu): đọc theo bảng mức ở trên, tối đa 20 từ. Ví dụ: *"Biển hiệu: Bún Bò Huế. Bún Đậu Mẹt."*
- `label` (nhãn đồ vật): đọc tên sản phẩm (chữ to nhất) trước, rồi đến các dòng HSD, NSX, dung tích. Không đọc bảng thành phần.
- `text` (văn bản, giấy tờ): đọc hết theo thứ tự tự nhiên, không lọc theo cỡ chữ.

**Điều kiện để một dòng được đọc (biển hiệu).** Xét lần lượt, gặp điều kiện nào trước thì dừng ở đó. Cột "Lý do" trên trang `/test` hiện đúng điều kiện đã quyết định:
1. Mức **Chi tiết** và chưa bật `detail` → bỏ qua.
2. *(chỉ khi dùng EasyOCR)* Khối nằm **chéo góc** so với chữ Chính, tức lệch cả theo chiều ngang lẫn chiều dọc → coi là biển bên cạnh, bỏ qua. Chữ nằm cùng hàng hoặc cùng cột với chữ Chính vẫn được đọc.
3. Không còn chữ nào đủ tin cậy: mức Chính cần ≥ `OCR_MIN_CONFIDENCE`, mức Phụ cần ≥ `OCR_SUB_MIN_CONFIDENCE` → bỏ qua (chữ mờ). Nếu mọi dòng Chính đều mờ thì nói "Có biển hiệu nhưng chữ chưa rõ".
4. **Giới hạn `OCR_SIGN_MAX_WORDS` từ**, không tính các ký hiệu như "-" hay "/": mức Chính luôn được đọc. Câu Phụ quá dài thì vẫn đọc các dòng đầu còn vừa. Các câu ngắn phía sau vẫn được xét tiếp.
5. Còn lại → **đọc**.

Các ngưỡng trên chỉnh được trong `server/.env` qua các biến `OCR_TIER_*`, `OCR_SUB_MIN_CONFIDENCE`, `OCR_SIGN_MAX_WORDS` và `OCR_LABEL_MAX_WORDS`.

Luồng trên ESP32-CAM: chụp ảnh → `POST` tới endpoint tương ứng với thao tác nút → đọc `audio_url` trong JSON → tải MP3 và phát qua loa.

### Trợ lý mô tả cảnh cho Single Click

Tạo khóa trong [Google AI Studio](https://aistudio.google.com/app/apikey), điền `GEMINI_API_KEY` trong `server/.env` rồi khởi động lại server. Mặc định `SCENE_AI_MODEL=gemini-3.5-flash-lite` và thời gian chờ là 12 giây (`SCENE_AI_TIMEOUT_SECONDS`). Server gửi ảnh JPEG đã thu nhỏ cùng gợi ý YOLO tới Gemini. Dùng khóa thuộc dự án Free Tier để chạy trong [hạn mức miễn phí của Google](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite).

Khi chạy trên mạng dùng chung, có thể giới hạn thiết bị được gọi API bằng `API_KEY` và header `X-API-Key` trên ESP32-CAM.

Mặc định `SCENE_AI_MEDIA_RESOLUTION=LOW` để giảm lượng dữ liệu ảnh Gemini xử lý. Nếu mô tả bỏ sót chi tiết nhỏ, đặt `SCENE_AI_MEDIA_RESOLUTION=MEDIUM` trong `server/.env` rồi khởi động lại. Thiết lập này chỉ áp dụng cho Gemini.

Prompt của trợ lý nằm ở [scene_assistant_prompt.md](server/app/services/scene_assistant_prompt.md). Trợ lý ưu tiên vật cản và hướng, dùng tối đa hai câu tiếng Việt ngắn, và diễn đạt mức độ không chắc chắn nếu ảnh khó đọc. `spoken_objects` liệt kê các phát hiện YOLO mà trợ lý đã nhắc tới. Nếu thiếu khóa hoặc dịch vụ AI lỗi, endpoint vẫn trả kết quả YOLO kèm `description_error` và thử tạo MP3 thông báo. `/api/detect` dùng mô tả cục bộ để kiểm thử.

---

## 6. Kiểm thử

1. **Swagger UI:** <http://localhost:8000/docs> — thử trực tiếp mọi endpoint.
2. **Trang kiểm thử:** <http://localhost:8000/test> — upload ảnh hoặc chụp webcam, chọn chế độ *Nhận diện vật thể* hoặc *Đọc chữ (OCR)*, xem kết quả và nghe giọng đọc. Ở chế độ OCR có thể chọn loại nội dung và bật *Đọc cả chữ nhỏ*. Bảng kết quả hiện mức chữ của từng dòng và dòng nào được đọc.
3. **Unit test cách đọc chữ** (không cần EasyOCR hay ảnh thật):
   ```bash
   pip install pytest
   cd server
   python -m pytest tests -v
   ```

---

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `Could not load YOLO model` | Kiểm tra `YOLO_MODEL_PATH` là file `*-worldv2.pt`; kiểm tra Internet cho lần tải đầu. |
| `/health` báo `ocr_loaded: false` | Lần đầu EasyOCR cần Internet để tải model vào `~/.EasyOCR/`. Xem log khởi động. Nếu GitHub trả lỗi 504, tải file qua GitHub API (`api.github.com/repos/JaidedAI/EasyOCR/releases/assets/<id>`) rồi đặt vào `~/.EasyOCR/model/`. Khi đã có `GEMINI_API_KEY`, chức năng đọc chữ vẫn chạy được mà không cần EasyOCR. |
| Câu đọc sai dấu hoặc dính chữ (`engine: easyocr`) | Điền `GEMINI_API_KEY` để dùng Gemini. Xem `engine_error` nếu Gemini vẫn bị bỏ qua. |
| `audio_error` trong phản hồi | Edge-TTS / gTTS cần Internet; kết quả nhận diện/OCR vẫn được trả về. |
| ESP32 không kết nối được | Máy chủ và ESP32 phải cùng mạng Wi-Fi; mở cổng 8000 trên Windows Firewall. |
| Chạy chậm trên GPU 4GB | Đặt `MAX_CONCURRENT_REQUESTS=1` hoặc dùng `DEVICE=cpu` cho ổn định. |
