import datetime
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Ensure backend directory and project root are in sys.path
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import efficientnet_b0

from config import settings
from database import init_db, close_db, check_database_connection, get_database
from auth import get_optional_user
from security import validate_file_magic_bytes, get_user_storage_dir, rate_limiter
from gradcam import generate_gradcam_overlay
from predict_video import predict_video
from video_utils import save_uploaded_video
from predict_audio import predict_audio

# Routers
from routers.auth_router import router as auth_router
from routers.scans_router import router as scans_router
from routers.audit_router import router as audit_router

# Model and device configuration
MODEL_PATH = settings.image_model_path
DEVICE = settings.device


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for database and service lifecycle."""
    # Startup
    init_db()
    yield
    # Shutdown
    close_db()


app = FastAPI(
    title="DeepGuard API",
    version="2.0.0",
    description="AI Deepfake Detection API with Multi-Media Analysis, Authentication, and Explainability",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount feature routers
app.include_router(auth_router)
app.include_router(scans_router)
app.include_router(audit_router)

# Image model preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Load existing image model
if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Image model not found: {MODEL_PATH}")

model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

classes = ["fake", "real"]

print("===================================")
print(" DeepGuard AI Engine Ready")
print(" Image Model   : Loaded (EfficientNet-B0)")
print(" Audio Model   : Ready (models/deepguard_audio.pth)")
print(" Compute Device:", DEVICE)
print(" Environment   :", settings.environment)
print("===================================")


@app.get("/")
def home():
    db_info = check_database_connection()
    return {
        "message": "DeepGuard API is Running",
        "model": "EfficientNet-B0",
        "status": "Ready",
        "audio_model": (
            "Available"
            if settings.audio_model_path.is_file()
            else "Not trained yet"
        ),
        "database": {
            "status": db_info["status"],
            "connected": db_info["connected"],
        },
    }


@app.get("/health")
def health_check():
    """
    Comprehensive health check reporting API readiness, database status, and models.
    Does not expose sensitive credentials, connection strings, or system paths.
    """
    db_info = check_database_connection()
    image_model_ready = MODEL_PATH.is_file()
    audio_model_ready = settings.audio_model_path.is_file()

    is_operational = image_model_ready
    overall_status = "healthy" if is_operational else "degraded"

    return {
        "status": overall_status,
        "api": "online",
        "database": {
            "status": db_info["status"],
            "connected": db_info["connected"],
            "message": db_info["message"],
            "database_name": db_info["database_name"],
        },
        "models": {
            "image_model": "loaded" if image_model_ready else "unavailable",
            "audio_model": "available" if audio_model_ready else "not_trained",
            "device": str(DEVICE),
        },
        "environment": settings.environment,
        "version": "2.0.0",
    }


def predict_image(image: Image.Image):
    image_rgb = image.convert("RGB")
    image_tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    pred_idx = prediction.item()
    pred_label = classes[pred_idx]
    conf_value = round(confidence.item() * 100, 2)

    # Generate Grad-CAM and forensic indicators
    heatmap_uri = None
    forensic_obs = []
    feature_wts = []
    try:
        heatmap_uri, forensic_obs, feature_wts = generate_gradcam_overlay(
            model, image_rgb, transform, DEVICE, pred_idx
        )
    except Exception as exc:
        print(f"Grad-CAM generation warning: {exc}")

    return {
        "prediction": pred_label,
        "confidence": conf_value,
        "heatmap": heatmap_uri,
        "gradcam": heatmap_uri,
        "forensic_observations": forensic_obs,
        "feature_weights": feature_wts,
        "media_type": "image",
    }


def save_scan_record(result_data: dict, filename: str, file_size: int, media_type: str, user: Optional[dict] = None) -> Optional[str]:
    """Helper to persist scan record to analyses collection if user is authenticated."""
    db = get_database()
    if db is None or not user:
        return None

    try:
        doc = {
            "user_id": user["id"],
            "username": user["username"],
            "filename": filename,
            "file_size": file_size,
            "media_type": media_type,
            "prediction": result_data.get("prediction"),
            "confidence": result_data.get("confidence"),
            "fake_frames": result_data.get("fake_frames"),
            "real_frames": result_data.get("real_frames"),
            "total_frames": result_data.get("total_frames"),
            "duration_sec": result_data.get("duration_sec"),
            "heatmap": result_data.get("heatmap"),
            "spectrogram": result_data.get("spectrogram"),
            "forensic_observations": result_data.get("forensic_observations"),
            "feature_weights": result_data.get("feature_weights"),
            "analyzed_at": datetime.datetime.now(datetime.timezone.utc),
        }
        res = db["analyses"].insert_one(doc)
        scan_id = str(res.inserted_id)

        db["audit_logs"].insert_one({
            "action": "SCAN_CREATED",
            "user_id": user["id"],
            "username": user["username"],
            "timestamp": doc["analyzed_at"],
            "details": f"Analyzed {media_type} '{filename}' with verdict {result_data.get('prediction')} ({result_data.get('confidence')}%)",
        })

        return scan_id
    except Exception as exc:
        print(f"Error persisting scan: {exc}")
        return None


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    header = await file.read(64)
    await file.seek(0)

    # Magic byte validation
    is_valid, mime_or_err = validate_file_magic_bytes(header, file.filename or "image.jpg")
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {mime_or_err}")

    try:
        contents = await file.read()
        if len(contents) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum upload size of {settings.max_upload_size_mb} MB.",
            )

        image = Image.open(file.file).convert("RGB")
        result = predict_image(image)

        # Attach metadata
        result["filename"] = file.filename
        result["file_size"] = len(contents)
        result["analyzed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Save scan record if user is authenticated
        scan_id = save_scan_record(result, file.filename or "image.jpg", len(contents), "image", current_user)
        if scan_id:
            result["scan_id"] = scan_id

        return result
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.") from exc
    finally:
        await file.close()


@app.post("/predict-video")
async def predict_video_api(
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    header = await file.read(64)
    await file.seek(0)

    is_valid, mime_or_err = validate_file_magic_bytes(header, file.filename or "video.mp4")
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid video format: {mime_or_err}")

    video_path = None
    try:
        video_path = save_uploaded_video(file)
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0

        if file_size > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Video exceeds maximum upload size of {settings.max_upload_size_mb} MB.",
            )

        result = predict_video(
            video_path,
            max_frames=settings.max_video_frames,
            frame_interval=settings.video_frame_interval,
        )

        result["filename"] = file.filename
        result["file_size"] = file_size
        result["analyzed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        scan_id = save_scan_record(result, file.filename or "video.mp4", file_size, "video", current_user)
        if scan_id:
            result["scan_id"] = scan_id

        return result
    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        await file.close()


@app.post("/predict-audio")
async def predict_audio_api(
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    header = await file.read(64)
    await file.seek(0)

    is_valid, mime_or_err = validate_file_magic_bytes(header, file.filename or "audio.wav")
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid audio format: {mime_or_err}")

    suffix = Path(file.filename or "").suffix.lower()
    temp_audio_path = None

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if len(contents) > settings.max_upload_size_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Audio exceeds maximum upload size of {settings.max_upload_size_mb} MB.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(contents)
            temp_audio_path = temp_audio.name

        try:
            result = predict_audio(temp_audio_path)
            result["filename"] = file.filename
            result["file_size"] = len(contents)
            result["analyzed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            scan_id = save_scan_record(result, file.filename or "audio.wav", len(contents), "audio", current_user)
            if scan_id:
                result["scan_id"] = scan_id

            return result
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not process this audio file: {exc}",
            ) from exc

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        await file.close()
