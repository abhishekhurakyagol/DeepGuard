import os
import time
from pathlib import Path
from typing import Optional, Tuple
from fastapi import HTTPException, UploadFile
from config import settings, PROJECT_ROOT

# Storage directories
UPLOAD_BASE_DIR = PROJECT_ROOT / "uploads"
UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Magic bytes definitions for robust MIME verification
MAGIC_NUMBERS = {
    # Images
    "jpeg": [b"\xFF\xD8\xFF"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"],  # Followed by WEBP at offset 8
    "bmp": [b"BM"],
    # Audio
    "wav": [b"RIFF"],  # Followed by WAVE at offset 8
    "mp3": [b"ID3", b"\xFF\xFB", b"\xFF\xF3", b"\xFF\xF2"],
    "flac": [b"fLaC"],
    "ogg": [b"OggS"],
    # Video
    "mp4": [b"ftyp", b"moov", b"mdat"],  # ftyp typically at offset 4
}


def validate_file_magic_bytes(header: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validate that the file header matches its claimed extension.
    Prevents file extension spoofing attacks.
    """
    ext = Path(filename).suffix.lower().lstrip(".")
    if not ext:
        return False, "Missing file extension"

    # Check JPEG
    if ext in ("jpg", "jpeg"):
        if any(header.startswith(magic) for magic in MAGIC_NUMBERS["jpeg"]):
            return True, "image/jpeg"
        return False, "Invalid JPEG header format"

    # Check PNG
    if ext == "png":
        if any(header.startswith(magic) for magic in MAGIC_NUMBERS["png"]):
            return True, "image/png"
        return False, "Invalid PNG header format"

    # Check WEBP
    if ext == "webp":
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP":
            return True, "image/webp"
        return False, "Invalid WEBP header format"

    # Check WAV
    if ext == "wav":
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WAVE":
            return True, "audio/wav"
        return False, "Invalid WAV header format"

    # Check MP3
    if ext == "mp3":
        if any(header.startswith(magic) for magic in MAGIC_NUMBERS["mp3"]):
            return True, "audio/mpeg"
        return False, "Invalid MP3 header format"

    # Check FLAC
    if ext == "flac":
        if header.startswith(b"fLaC"):
            return True, "audio/flac"
        return False, "Invalid FLAC header format"

    # Check OGG
    if ext == "ogg":
        if header.startswith(b"OggS"):
            return True, "audio/ogg"
        return False, "Invalid OGG header format"

    # Check MP4 / MOV / M4A (ISO media files)
    if ext in ("mp4", "mov", "m4a"):
        if len(header) >= 8 and (b"ftyp" in header[:16] or b"moov" in header[:32] or b"mdat" in header[:32]):
            media_type = "audio/mp4" if ext == "m4a" else "video/mp4"
            return True, media_type
        return False, f"Invalid {ext.upper()} media container format"

    # Check AVI / MKV / WEBM
    if ext == "avi":
        if header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"AVI ":
            return True, "video/x-msvideo"
        return False, "Invalid AVI header format"

    if ext in ("mkv", "webm"):
        if header.startswith(b"\x1a\x45\xdf\xa3"):  # Matroska / EBML header
            return True, f"video/{ext}"
        return False, f"Invalid {ext.upper()} container format"

    return False, f"Unsupported file type: .{ext}"


def get_user_storage_dir(user_id: Optional[str] = None) -> Path:
    """Return isolated storage directory for a user, or guest directory."""
    sub_dir = user_id if user_id else "guests"
    path = UPLOAD_BASE_DIR / sub_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


class InMemoryRateLimiter:
    """Simple sliding-window in-memory rate limiter per IP address."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}

    def check(self, key: str) -> bool:
        now = time.time()
        timestamps = self.requests.get(key, [])
        # Prune expired timestamps
        timestamps = [t for t in timestamps if now - t < self.window_seconds]
        if len(timestamps) >= self.max_requests:
            return False
        timestamps.append(now)
        self.requests[key] = timestamps
        return True


rate_limiter = InMemoryRateLimiter(max_requests=120, window_seconds=60)
