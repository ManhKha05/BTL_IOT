import asyncio
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple
import edge_tts
from fastapi import HTTPException, status

from app.core.config import settings
from app.core.logging import logger

# Regex to strictly validate hex UUID to prevent any path traversal attempt
UUID_REGEX = re.compile(r"^[a-fA-F0-9]{32}$")


def _prepend_silence_frames(file_path: Path, duration_seconds: float = 0.8) -> None:
    """
    Chèn trực tiếp các khung âm thanh im lặng (silent MP3 frames) vào đầu file MP3.
    Đảm bảo 100% phần cứng ESP32 / MAX98357A có thời gian bật bộ khuếch đại và đệm buffer,
    không phụ thuộc vào việc engine TTS có bỏ qua dấu câu hay không.
    """
    try:
        data = file_path.read_bytes()
        if len(data) < 4:
            return

        # Tìm sync word của MP3 (11 bit 1 liên tiếp)
        idx = 0
        while idx < len(data) - 4:
            if data[idx] == 0xFF and (data[idx + 1] & 0xE0) == 0xE0:
                break
            idx += 1

        if idx >= len(data) - 4:
            return

        b1, b2 = data[idx + 1], data[idx + 2]
        ver = (b1 >> 3) & 3
        layer = (b1 >> 1) & 3
        bitrate_idx = (b2 >> 4) & 15
        sr_idx = (b2 >> 2) & 3
        padding = (b2 >> 1) & 1

        # MPEG 1 Layer III
        if ver == 3 and layer == 1:
            sr_table = [44100, 48000, 32000]
            br_table = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
            if sr_idx < len(sr_table) and bitrate_idx < len(br_table):
                sr = sr_table[sr_idx]
                br = br_table[bitrate_idx] * 1000
                frame_len = int(144 * br / sr) + padding
                samples_per_frame = 1152
            else:
                return
        # MPEG 2 / 2.5 Layer III (gTTS phổ biến)
        elif ver in (0, 2) and layer == 1:
            sr_table = [22050, 24000, 16000] if ver == 2 else [11025, 12000, 8000]
            br_table = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
            if sr_idx < len(sr_table) and bitrate_idx < len(br_table):
                sr = sr_table[sr_idx]
                br = br_table[bitrate_idx] * 1000
                frame_len = int(72 * br / sr) + padding
                samples_per_frame = 576
            else:
                return
        else:
            return

        frame_ms = samples_per_frame / sr
        num_frames = max(1, int(duration_seconds / frame_ms))

        # Tạo frame im lặng: giữ nguyên header 4 bytes hợp lệ, xoá sạch payload về 0
        silence_frame = bytearray(data[idx:idx + frame_len])
        for k in range(4, len(silence_frame)):
            silence_frame[k] = 0

        # Ghép các frame im lặng vào trước toàn bộ dữ liệu âm thanh
        new_data = bytes(silence_frame) * num_frames + data
        file_path.write_bytes(new_data)
        logger.info(f"Prepended {num_frames} silence MP3 frames (~{duration_seconds}s) for hardware anti-clip.")
    except Exception as e:
        logger.warning(f"Could not prepend silence frames to MP3: {e}")


class TTSService:
    """
    Text-to-Speech service using Microsoft Edge-TTS:
    - Generates high-quality Vietnamese audio files (MP3) asynchronously.
    - Manages concurrency using asyncio.Semaphore to prevent network/CPU spikes.
    - Generates atomic temporary files and cleans up partial files on failure.
    - Implements TTL-based automatic garbage collection of expired audio.
    - Architecture designed for future plug-and-play TTS engine replacements.
    """

    def __init__(self):
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.output_dir: Path = settings.AUDIO_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily initialize semaphore inside the active event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TTS)
        return self._semaphore

    async def generate_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        engine: Optional[str] = None
    ) -> Tuple[str, Path, int, float]:
        """
        Chuyển đổi văn bản tiếng Việt thành file MP3 với GIỌNG NỮ MIỀN BẮC (mặc định chuẩn).

        Args:
            text: Nội dung văn bản tiếng Việt cần đọc (tối đa 500 ký tự).
            voice: Giọng Edge-TTS (chỉ dùng khi engine="edge"), mặc định vi-VN-HoaiMyNeural.
            rate: Tốc độ đọc Edge-TTS (ví dụ "+0%"), mặc định settings.TTS_RATE.
            engine: None/"gtts" giữ thứ tự mặc định (gTTS trước, Edge-TTS dự phòng);
                    "edge" dùng Edge-TTS trước, gTTS làm dự phòng.

        Returns:
            Tuple: (audio_id, file_path, file_size_bytes, duration_ms)
        """
        import unicodedata
        cleaned_text = unicodedata.normalize("NFC", text.strip())
        if not cleaned_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nội dung văn bản không được để trống."
            )

        if len(cleaned_text) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Độ dài văn bản ({len(cleaned_text)} ký tự) vượt quá giới hạn cho phép (500 ký tự)."
            )

        # Thêm khoảng lặng nhẹ ở đầu câu (bằng dấu chấm lửng / dấu phẩy ngắt quãng)
        # Giúp DAC/IC khuếch đại (MAX98357A) và buffer I2S trên ESP32 kịp khởi động, không bị nuốt 2-3 từ đầu.
        spoken_text = f"... , {cleaned_text}"

        audio_id = uuid.uuid4().hex
        final_file = self.output_dir / f"{audio_id}.mp3"
        temp_file = self.output_dir / f"{audio_id}.tmp.mp3"

        start_time = time.perf_counter()
        semaphore = self._get_semaphore()

        async with semaphore:
            # 0. Edge-TTS ưu tiên khi được yêu cầu rõ ràng (engine="edge")
            if (engine or "").lower() == "edge":
                edge_voice = voice or "vi-VN-HoaiMyNeural"
                edge_rate = rate or settings.TTS_RATE
                # Dịch vụ Edge-TTS đôi khi trả "No audio was received" ngẫu nhiên -> thử lại vài lần
                edge_attempts = 3
                for attempt in range(edge_attempts):
                    try:
                        logger.info(
                            f"Synthesizing Edge-TTS voice '{edge_voice}' (attempt {attempt + 1}/{edge_attempts}) "
                            f"for audio_id={audio_id} ({len(spoken_text)} chars)..."
                        )
                        communicate = edge_tts.Communicate(text=spoken_text, voice=edge_voice, rate=edge_rate)
                        await asyncio.wait_for(communicate.save(str(temp_file)), timeout=settings.TTS_TIMEOUT_SECONDS)
                        if not temp_file.exists() or temp_file.stat().st_size == 0:
                            raise RuntimeError("Tệp âm thanh Edge-TTS sinh ra bị rỗng.")
                        temp_file.replace(final_file)
                        _prepend_silence_frames(final_file, duration_seconds=0.8)
                        duration_ms = (time.perf_counter() - start_time) * 1000.0
                        file_size = final_file.stat().st_size
                        logger.info(f"Edge-TTS speech created: audio_id={audio_id}, size={file_size} bytes in {duration_ms:.1f}ms.")
                        return audio_id, final_file, file_size, round(duration_ms, 2)
                    except Exception as e:
                        if temp_file.exists():
                            try:
                                temp_file.unlink()
                            except OSError:
                                pass
                        if attempt < edge_attempts - 1:
                            await asyncio.sleep(0.3 * (attempt + 1))
                        else:
                            logger.warning(f"Edge-TTS failed after {edge_attempts} attempts ({e}). Falling back to gTTS...")

            # 1. ƯU TIÊN HÀNG ĐẦU: Giọng Nữ Miền Bắc (Chị Google - Chuẩn Bắc)
            max_retries = 3
            last_err = None

            for attempt in range(max_retries):
                try:
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except OSError:
                            pass

                    logger.info(
                        f"Synthesizing Northern female voice (attempt {attempt + 1}/{max_retries}) "
                        f"for audio_id={audio_id} ({len(spoken_text)} chars)..."
                    )

                    from gtts import gTTS
                    tts = gTTS(text=spoken_text, lang="vi", slow=False)
                    await asyncio.wait_for(
                        asyncio.to_thread(tts.save, str(temp_file)),
                        timeout=settings.TTS_TIMEOUT_SECONDS
                    )

                    if not temp_file.exists() or temp_file.stat().st_size == 0:
                        raise RuntimeError("Tệp âm thanh sinh ra bị rỗng.")

                    temp_file.replace(final_file)
                    _prepend_silence_frames(final_file, duration_seconds=0.8)
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    file_size = final_file.stat().st_size

                    logger.info(
                        f"Northern female speech created: audio_id={audio_id}, "
                        f"size={file_size} bytes in {duration_ms:.1f}ms."
                    )
                    return audio_id, final_file, file_size, round(duration_ms, 2)

                except Exception as e:
                    last_err = e
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                        except OSError:
                            pass
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.4 * (attempt + 1))

            # 2. DỰ PHÒNG: Nếu dịch vụ chính mất kết nối, thử dự phòng với Edge-TTS
            try:
                logger.warning(f"Primary Northern female TTS failed ({last_err}). Trying fallback Edge-TTS...")
                communicate = edge_tts.Communicate(text=cleaned_text, voice="vi-VN-NamMinhNeural")
                await asyncio.wait_for(communicate.save(str(temp_file)), timeout=settings.TTS_TIMEOUT_SECONDS)
                if temp_file.exists() and temp_file.stat().st_size > 0:
                    temp_file.replace(final_file)
                    _prepend_silence_frames(final_file, duration_seconds=0.8)
                    duration_ms = (time.perf_counter() - start_time) * 1000.0
                    return audio_id, final_file, final_file.stat().st_size, round(duration_ms, 2)
            except Exception:
                pass

            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Không thể kết nối đến dịch vụ giọng nói: {str(last_err)}"
            )

    def get_audio_file(self, audio_id: str) -> Optional[Path]:
        """
        Retrieve audio file Path for a given audio_id.
        Enforces strict alphanumeric validation to prevent directory traversal attacks.
        """
        # Strict validation: audio_id must be exactly 32-character hex
        if not UUID_REGEX.match(audio_id):
            return None

        target_file = self.output_dir / f"{audio_id}.mp3"
        try:
            # Ensure path is strictly inside output_dir
            resolved = target_file.resolve()
            if resolved.parent != self.output_dir.resolve():
                return None
            if resolved.is_file() and resolved.exists():
                return resolved
        except (ValueError, OSError):
            return None

        return None

    def cleanup_expired_files(self, ttl_minutes: Optional[int] = None) -> int:
        """
        Automatically remove audio files that have exceeded the TTL limit.
        Does not delete active temporary files (.tmp.mp3) or files modified within TTL.

        Returns:
            Number of removed files.
        """
        ttl = ttl_minutes if ttl_minutes is not None else settings.AUDIO_TTL_MINUTES
        cutoff_time = time.time() - (ttl * 60)
        deleted_count = 0

        try:
            for item in self.output_dir.iterdir():
                if item.is_file() and item.suffix.lower() == ".mp3":
                    # Skip temporary writing files
                    if item.name.endswith(".tmp.mp3"):
                        # If a .tmp.mp3 is older than 5 minutes, it's an abandoned artifact from a crash
                        if item.stat().st_mtime < (time.time() - 300):
                            try:
                                item.unlink()
                                deleted_count += 1
                            except OSError:
                                pass
                        continue

                    # Normal mp3 check
                    if item.stat().st_mtime < cutoff_time:
                        try:
                            item.unlink()
                            deleted_count += 1
                            logger.info(f"Cleaned up expired audio file: {item.name}")
                        except OSError as e:
                            logger.warning(f"Could not delete expired file {item.name}: {e}")
        except Exception as e:
            logger.error(f"Error during audio cleanup routine: {e}")

        return deleted_count


# Global singleton instance
tts_service = TTSService()
