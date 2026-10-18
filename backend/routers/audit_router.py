import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File

from auth import get_current_user, get_optional_user
from database import get_database

router = APIRouter(prefix="/api", tags=["Audit & Multimodal"])


@router.get("/audit")
def list_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve security and operational audit trail.
    Users view their own activity; administrators can view all events.
    """
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    query: Dict[str, Any] = {}
    if current_user.get("role") != "admin":
        query["user_id"] = current_user["id"]

    cursor = db["audit_logs"].find(query).sort("timestamp", -1).limit(limit)

    logs = []
    for doc in cursor:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
        logs.append(doc)

    return {"logs": logs, "count": len(logs)}


@router.post("/predict-multimodal")
async def predict_multimodal_api(
    file: UploadFile = File(...),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Multimodal video + audio deepfake analysis endpoint.
    Reports clear unconfigured status rather than fabricating predictions.
    """
    return {
        "status": "not_configured",
        "prediction": "inconclusive",
        "confidence": 0.0,
        "message": (
            "Dedicated cross-modal fusion detector (video + audio joint feature model) "
            "is not configured in this environment. Please analyze video and audio separately "
            "using /predict-video and /predict-audio."
        ),
        "visual_pipeline_available": True,
        "audio_pipeline_available": True,
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
    }
