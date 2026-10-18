from pathlib import Path
import os
from typing import List
from dotenv import load_dotenv
import torch

# Locate project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file from PROJECT_ROOT or backend directory
env_path = PROJECT_ROOT / ".env"
if env_path.is_file():
    load_dotenv(dotenv_path=env_path)
else:
    backend_env = PROJECT_ROOT / "backend" / ".env"
    if backend_env.is_file():
        load_dotenv(dotenv_path=backend_env)


class Settings:
    """Centralized DeepGuard application configuration."""

    def __init__(self):
        # Server
        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.environment: str = os.getenv("ENVIRONMENT", "development")

        # CORS Origins
        raw_origins = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        self.cors_origins: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]

        # MongoDB Atlas
        self.mongodb_uri: str = os.getenv("MONGODB_URI", "").strip()
        self.mongodb_db_name: str = os.getenv("MONGODB_DB_NAME", "deepguard").strip()
        self.mongodb_timeout_ms: int = int(os.getenv("MONGODB_TIMEOUT_MS", "3000"))

        # Security & JWT
        self.jwt_secret: str = os.getenv(
            "JWT_SECRET",
            "deepguard-local-dev-secret-key-32chars-minimum!",
        )
        self.jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
        self.jwt_expire_minutes: int = int(
            os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
        )

        # Upload limits
        self.max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "100"))
        self.max_video_frames: int = int(os.getenv("MAX_VIDEO_FRAMES", "60"))
        self.video_frame_interval: int = int(os.getenv("VIDEO_FRAME_INTERVAL", "15"))

        # Model Paths
        image_model_env = os.getenv("IMAGE_MODEL_PATH", "models/deepguard_v1.pth")
        self.image_model_path: Path = (
            Path(image_model_env)
            if Path(image_model_env).is_absolute()
            else PROJECT_ROOT / image_model_env
        )

        audio_model_env = os.getenv("AUDIO_MODEL_PATH", "models/deepguard_audio.pth")
        self.audio_model_path: Path = (
            Path(audio_model_env)
            if Path(audio_model_env).is_absolute()
            else PROJECT_ROOT / audio_model_env
        )

        # Hardware acceleration
        device_env = os.getenv("DEVICE", "auto").lower()
        if device_env == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_env == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_mongodb_configured(self) -> bool:
        """Check whether a realistic MongoDB connection string is provided."""
        if not self.mongodb_uri:
            return False
        # Discard default placeholder strings
        if "<username>" in self.mongodb_uri or "<password>" in self.mongodb_uri:
            return False
        return self.mongodb_uri.startswith("mongodb://") or self.mongodb_uri.startswith("mongodb+srv://")

    @property
    def masked_mongodb_uri(self) -> str:
        """Return a sanitized MongoDB URI safe for display (hides credentials)."""
        if not self.is_mongodb_configured:
            return "not_configured"
        try:
            # Mask user:pass if present
            if "@" in self.mongodb_uri:
                prefix, remainder = self.mongodb_uri.split("@", 1)
                protocol = prefix.split("://")[0]
                return f"{protocol}://***:***@{remainder.split('?')[0]}"
            return self.mongodb_uri.split("?")[0]
        except Exception:
            return "configured (masked)"


settings = Settings()
