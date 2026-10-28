# BTL IoT - Assistive AI Server for Visually Impaired

Hệ thống AI Server hỗ trợ người khiếm thị nhận diện vật thể và cảnh báo âm thanh thời gian thực qua ESP32-CAM, xây dựng trên nền tảng **FastAPI**, **Ultralytics YOLO (YOLO-World / YOLOv8)** và **Microsoft Edge-TTS / gTTS**.

---

## 1. Yêu cầu môi trường

- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04+ hoặc macOS.
- **Python**: Khuyến nghị phiên bản **Python 3.11** (đã kiểm thử ổn định trên Python 3.11.8).
- **GPU (Tùy chọn)**: Hỗ trợ NVIDIA CUDA 12.1 hoặc chạy mượt mà trên CPU.

---

## 2. Hướng dẫn cài đặt và thiết lập

### Bước 1: Clone dự án
```bash
git clone <URL_REPOSITORY_CUA_BAN>
cd BTL_IOT_AI
```

### Bước 2: Tạo và kích hoạt môi trường ảo (.venv)
- **Trên Windows (Command Prompt / PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```

- **Trên Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Bước 3: Cài đặt thư viện phụ thuộc
Cài đặt danh sách thư viện từ file `requirements.txt` tại thư mục gốc:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Lưu ý về PyTorch:**
> File `requirements.txt` đã cấu hình sẵn index URL cho CUDA 12.1 (`--extra-index-url https://download.pytorch.org/whl/cu121`).
> Nếu bạn chỉ muốn chạy trên CPU thuần túy, có thể cài đặt bản PyTorch CPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### Bước 4: Thiết lập file cấu hình môi trường (.env)
Tạo file `.env` bên trong thư mục `server/` từ file mẫu `server/.env.example`:
- **Windows PowerShell:**
  ```powershell
  Copy-Item server\.env.example server\.env
  ```
- **Linux / macOS:**
  ```bash
  cp server/.env.example server/.env
  ```
*(Có thể tùy chỉnh IP, Port, TTS voice trong file `server/.env` nếu cần).*

---

## 3. Mô hình AI (YOLO Weights)

Hệ thống sử dụng mô hình nhận diện vật thể và chướng ngại vật đường phố:
- Mặc định: `server/models/yolov8s-worldv2.pt` (hoặc `yolov8n.pt`).
- **Tự động tải:** Thư viện `ultralytics` sẽ **tự động tải** trọng số mô hình khi bạn khởi động server lần đầu tiên nếu file chưa có sẵn.
- **Tải thủ công (nếu cần):** Tải file trọng số [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt) hoặc [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt) và đặt vào thư mục `server/models/`.

---

## 4. Khởi chạy Server

Đảm bảo môi trường `.venv` đang được kích hoạt:

### Cách 1: Chạy từ thư mục gốc
```bash
python server/main.py
```

### Cách 2: Chuyển vào thư mục server
```bash
cd server
python main.py
```

### Cách 3: Chạy trực tiếp qua Uvicorn (kèm Hot-reload)
```bash
cd server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 5. Kiểm thử hệ thống

Sau khi server khởi động thành công:
1. **API Swagger UI:** Truy cập [http://localhost:8000/docs](http://localhost:8000/docs) để xem tài liệu API và test trực tiếp các endpoint.
2. **Giao diện Web kiểm thử trực quan:** Truy cập [http://localhost:8000/test](http://localhost:8000/test) để upload ảnh kiểm tra trực tiếp bounding box nhận diện và nghe phát âm thanh TTS tiếng Việt.
3. **Endpoint chính dành cho ESP32-CAM:**
   - `POST /api/process/detect` (Nhận diện vật thể, tạo câu mô tả tiếng Việt và sinh file âm thanh phản hồi).
   - `GET /api/audio/{audio_id}` (Tải luồng âm thanh MP3).
