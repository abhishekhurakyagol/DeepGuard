import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings, Settings
from database import check_database_connection
from app import app


@pytest.fixture
def client():
    """Create FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client


def test_settings_initialization():
    """Verify settings load cleanly with valid defaults."""
    assert isinstance(settings.cors_origins, list)
    assert len(settings.cors_origins) > 0
    assert settings.image_model_path.is_file(), "Image model file must exist"
    assert settings.audio_model_path.is_file(), "Audio model file must exist"
    assert settings.max_upload_size_mb >= 10
    assert settings.environment in ("development", "production", "test")


def test_settings_masked_uri():
    """Ensure sensitive credentials in MongoDB URI are never leaked."""
    test_settings = Settings()
    # Test with standard URI containing password
    test_settings.mongodb_uri = "mongodb+srv://admin:SuperSecretPass123@mycluster.mongodb.net/deepguard?retryWrites=true"
    masked = test_settings.masked_mongodb_uri
    assert "SuperSecretPass123" not in masked
    assert "admin" not in masked
    assert "***:***" in masked

    # Test with empty/unconfigured
    test_settings.mongodb_uri = ""
    assert test_settings.masked_mongodb_uri == "not_configured"


def test_home_endpoint(client):
    """Test the base GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "DeepGuard API is Running"
    assert data["model"] == "EfficientNet-B0"
    assert "status" in data
    assert "audio_model" in data
    assert "database" in data
    assert "password" not in response.text
    assert "mongodb+srv" not in response.text


def test_health_endpoint(client):
    """Test the comprehensive GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["api"] == "online"
    assert data["status"] in ("healthy", "degraded")
    assert "database" in data
    assert "status" in data["database"]
    assert "connected" in data["database"]
    assert "message" in data["database"]

    assert "models" in data
    assert data["models"]["image_model"] == "loaded"
    assert data["models"]["audio_model"] == "available"
    assert "device" in data["models"]

    # Security check: verify no secrets leaked
    assert "password" not in response.text
    assert "mongodb+srv" not in response.text
    assert "secret" not in response.text.lower() or "secret" not in data


def test_database_graceful_handling_when_unconfigured():
    """Verify check_database_connection behaves safely when URI is missing or placeholder."""
    # Ensure current state without credentials returns safe dict without throwing
    res = check_database_connection()
    assert isinstance(res, dict)
    assert "status" in res
    assert "connected" in res
    assert res["status"] in ("not_configured", "connected", "disconnected")


def test_database_graceful_handling_on_invalid_uri(monkeypatch):
    """Verify that attempting to connect to an unreachable cluster fails gracefully without crash."""
    from database import check_database_connection
    import config

    monkeypatch.setattr(config.settings, "mongodb_uri", "mongodb://127.0.0.1:27019/deepguard")
    monkeypatch.setattr(config.settings, "mongodb_timeout_ms", 500)

    res = check_database_connection()
    assert isinstance(res, dict)
    assert res["connected"] is False
    assert res["status"] in ("disconnected", "not_configured")
    assert "timed out" in res["message"].lower() or "error" in res["message"].lower()


def test_model_preservation():
    """Verify that trained model weights and checkpoints remain intact and unmodified."""
    image_model_path = PROJECT_ROOT / "models" / "deepguard_v1.pth"
    audio_model_path = PROJECT_ROOT / "models" / "deepguard_audio.pth"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "latest_model.pth"

    assert image_model_path.is_file(), "deepguard_v1.pth must be preserved"
    assert audio_model_path.is_file(), "deepguard_audio.pth must be preserved"
    assert checkpoint_path.is_file(), "latest_model.pth checkpoint must be preserved"

    # Verify file sizes have not been wiped or corrupted
    assert image_model_path.stat().st_size > 15_000_000, "deepguard_v1.pth size sanity check"
    assert audio_model_path.stat().st_size > 15_000_000, "deepguard_audio.pth size sanity check"
    assert checkpoint_path.stat().st_size > 40_000_000, "latest_model.pth size sanity check"
