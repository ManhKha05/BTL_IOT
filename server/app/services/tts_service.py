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
        rate: Optional[str] = None
    ) -> Tuple[str, Path, int, float]:
        """
        Chuyển đổi văn bản tiếng Việt thành file MP3 với GIỌNG NỮ MIỀN BẮC (mặc định chuẩn).

        Args:
            text: Nội dung văn bản tiếng Việt cần đọc (tối đa 500 ký tự).

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

        audio_id = uuid.uuid4().hex
        final_file = self.output_dir / f"{audio_id}.mp3"
        temp_file = self.output_dir / f"{audio_id}.tmp.mp3"

        start_time = time.perf_counter()
        semaphore = self._get_semaphore()

        async with semaphore:
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
                        f"for audio_id={audio_id} ({len(cleaned_text)} chars)..."
                    )

                    from gtts import gTTS
                    tts = gTTS(text=cleaned_text, lang="vi", slow=False)
                    await asyncio.wait_for(
                        asyncio.to_thread(tts.save, str(temp_file)),
                        timeout=settings.TTS_TIMEOUT_SECONDS
                    )

                    if not temp_file.exists() or temp_file.stat().st_size == 0:
                        raise RuntimeError("Tệp âm thanh sinh ra bị rỗng.")

                    temp_file.replace(final_file)
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
