import io
import sys
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure backend is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import app
from database import get_database


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    """Register and login a temporary test user, return Authorization headers."""
    reg_payload = {
        "username": "tester_deepguard",
        "email": "tester@example.com",
        "password": "Password123!",
        "full_name": "DeepGuard Test User",
    }
    # Ensure clean state for test user
    db = get_database()
    if db is not None:
        db["users"].delete_many({"username": "tester_deepguard"})
        db["analyses"].delete_many({"username": "tester_deepguard"})

    reg_res = client.post("/api/auth/register", json=reg_payload)
    if reg_res.status_code == 201:
        token = reg_res.json()["token"]
    else:
        # User might already exist, login instead
        login_res = client.post("/api/auth/login", json={"identifier": "tester_deepguard", "password": "Password123!"})
        assert login_res.status_code == 200
        token = login_res.json()["token"]

    return {"Authorization": f"Bearer {token}"}


def test_health_check_operational(client):
    """Test health endpoint reports healthy status and local database."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["api"] == "online"
    assert data["models"]["image_model"] == "loaded"
    assert data["models"]["audio_model"] == "available"
    assert "database" in data


def test_auth_registration_and_login(client):
    """Test user registration, duplicate prevention, and login."""
    unique_user = "user_test_99"
    unique_email = "user99@example.com"

    db = get_database()
    if db is not None:
        db["users"].delete_many({"username": unique_user})

    reg_res = client.post("/api/auth/register", json={
        "username": unique_user,
        "email": unique_email,
        "password": "SecurePassword123!",
    })
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert "token" in reg_data
    assert reg_data["user"]["username"] == unique_user

    # Duplicate registration should be rejected
    dup_res = client.post("/api/auth/register", json={
        "username": unique_user,
        "email": unique_email,
        "password": "SecurePassword123!",
    })
    assert dup_res.status_code == 400

    # Successful login
    login_res = client.post("/api/auth/login", json={
        "identifier": unique_user,
        "password": "SecurePassword123!",
    })
    assert login_res.status_code == 200
    assert "token" in login_res.json()

    # Wrong password rejected
    bad_login = client.post("/api/auth/login", json={
        "identifier": unique_user,
        "password": "WrongPassword!",
    })
    assert bad_login.status_code == 401


def test_auth_me_endpoint(client, auth_headers):
    """Test protected /api/auth/me profile endpoint."""
    res = client.get("/api/auth/me", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "tester_deepguard"
    assert "email" in data


def test_image_prediction_with_gradcam(client, auth_headers):
    """Test real image detection on person.png with Grad-CAM and persistence."""
    image_path = PROJECT_ROOT / "test_images" / "person.png"
    assert image_path.is_file()

    with open(image_path, "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("person.png", f, "image/png")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("real", "fake")
    assert isinstance(data["confidence"], (int, float))
    assert data["heatmap"] is not None
    assert data["heatmap"].startswith("data:image/png;base64,")
    assert len(data["forensic_observations"]) > 0
    assert len(data["feature_weights"]) > 0
    assert "scan_id" in data


def test_video_prediction_with_timestamps(client, auth_headers):
    """Test video prediction on sample.mp4 with frame extraction and timeline."""
    video_path = PROJECT_ROOT / "sample.mp4"
    assert video_path.is_file()

    with open(video_path, "rb") as f:
        response = client.post(
            "/predict-video",
            files={"file": ("sample.mp4", f, "video/mp4")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("real", "fake")
    assert data["total_frames"] > 0
    assert "timeline" in data
    assert len(data["timeline"]) > 0
    assert "timestamp_sec" in data["timeline"][0]
    assert "scan_id" in data


def test_audio_prediction_with_spectrogram(client, auth_headers):
    """Test audio prediction on 0.wav with mel-spectrogram and waveform envelope."""
    audio_path = PROJECT_ROOT / "datasets" / "audio" / "fake" / "0.wav"
    assert audio_path.is_file()

    with open(audio_path, "rb") as f:
        response = client.post(
            "/predict-audio",
            files={"file": ("0.wav", f, "audio/wav")},
            headers=auth_headers,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "fake"
    assert data["spectrogram"] is not None
    assert data["spectrogram"].startswith("data:image/png;base64,")
    assert len(data["waveform"]) == 100
    assert data["duration_sec"] > 0
    assert "scan_id" in data


def test_scan_history_and_deletion(client, auth_headers):
    """Test scan history listing, filtering, and deletion."""
    # Create a scan for this test
    image_path = PROJECT_ROOT / "test_images" / "person.png"
    with open(image_path, "rb") as f:
        upload_res = client.post(
            "/predict",
            files={"file": ("person.png", f, "image/png")},
            headers=auth_headers,
        )
    assert upload_res.status_code == 200

    list_res = client.get("/api/scans", headers=auth_headers)
    assert list_res.status_code == 200
    data = list_res.json()
    assert "scans" in data
    assert data["total"] >= 1

    first_scan = data["scans"][0]
    scan_id = first_scan["id"]

    # Retrieve specific scan details
    detail_res = client.get(f"/api/scans/{scan_id}", headers=auth_headers)
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == scan_id

    # Delete scan
    del_res = client.delete(f"/api/scans/{scan_id}", headers=auth_headers)
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True


def test_multimodal_unconfigured_status(client):
    """Verify multimodal analysis reports honest unconfigured status without fake predictions."""
    fake_file = io.BytesIO(b"fake media content for test")
    res = client.post(
        "/api/predict-multimodal",
        files={"file": ("sample.mp4", fake_file, "video/mp4")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "not_configured"
    assert data["prediction"] == "inconclusive"


def test_audit_log_viewer(client, auth_headers):
    """Verify audit log captures events and can be retrieved by authenticated user."""
    res = client.get("/api/audit", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "logs" in data
    assert len(data["logs"]) > 0
