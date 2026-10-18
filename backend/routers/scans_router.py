import datetime
import math
from pathlib import Path
from typing import Optional, List, Dict, Any
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends, Query, status

from auth import get_current_user
from database import get_database
from audit import log_audit_event

router = APIRouter(prefix="/api/scans", tags=["Scan History"])


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to convert MongoDB BSON ObjectId to string."""
    if not doc:
        return {}
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc


@router.get("")
def list_scans(
    media_type: Optional[str] = Query(None, description="Filter by media type (image, video, audio)"),
    verdict: Optional[str] = Query(None, description="Filter by verdict (real, fake)"),
    search: Optional[str] = Query(None, description="Search by filename"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    current_user: dict = Depends(get_current_user),
):
    """
    List historical analysis scans for the authenticated user with filtering and pagination.
    Enforces user isolation and data ownership.
    """
    db = get_database()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is unavailable.",
        )

    # Base query restricted to current user's scans (or all if admin)
    query: Dict[str, Any] = {}
    if current_user.get("role") != "admin":
        query["user_id"] = current_user["id"]

    if media_type:
        query["media_type"] = media_type.lower()

    if verdict:
        query["prediction"] = verdict.lower()

    if search:
        query["filename"] = {"$regex": search.strip(), "$options": "i"}

    total_count = db["analyses"].count_documents(query)
    total_pages = max(1, math.ceil(total_count / limit))
    skip = (page - 1) * limit

    # Project fields, omitting huge raw base64 images from list view for fast network transfer
    projection = {
        "heatmap": 0,
        "spectrogram": 0,
    }

    cursor = db["analyses"].find(query, projection).sort("analyzed_at", -1).skip(skip).limit(limit)
    items = [serialize_doc(doc) for doc in cursor]

    return {
        "scans": items,
        "total": total_count,
        "page": page,
        "limit": limit,
        "pages": total_pages,
    }


@router.get("/{scan_id}")
def get_scan_details(
    scan_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve full scan record including explainability visualizations.
    Enforces strict user ownership verification.
    """
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    try:
        obj_id = ObjectId(scan_id)
        scan = db["analyses"].find_one({"_id": obj_id})
    except Exception:
        scan = db["analyses"].find_one({"_id": scan_id})

    if not scan:
        raise HTTPException(status_code=404, detail="Scan record not found.")

    # Ownership check
    if current_user.get("role") != "admin" and scan.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this analysis record.",
        )

    return serialize_doc(scan)


@router.delete("/{scan_id}")
def delete_scan(
    scan_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Delete a scan record and its associated stored file.
    Enforces strict user ownership verification.
    """
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    try:
        obj_id = ObjectId(scan_id)
        scan = db["analyses"].find_one({"_id": obj_id})
    except Exception:
        scan = db["analyses"].find_one({"_id": scan_id})

    if not scan:
        raise HTTPException(status_code=404, detail="Scan record not found.")

    # Ownership check
    if current_user.get("role") != "admin" and scan.get("user_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this analysis record.",
        )

    # Delete associated file if stored on disk
    file_path = scan.get("storage_path")
    if file_path:
        try:
            path = Path(file_path)
            if path.is_file():
                path.unlink(missing_ok=True)
        except Exception:
            pass

    # Delete record from database
    db["analyses"].delete_one({"_id": scan["_id"]})

    # Record in audit log
    log_audit_event(
        action="SCAN_DELETED",
        user_id=current_user["id"],
        username=current_user["username"],
        details=f"Scan {scan_id} ({scan.get('filename')}) deleted by owner",
    )

    return {
        "success": True,
        "message": "Scan record and associated files have been permanently deleted.",
    }


@router.delete("/user/purge")
def purge_user_data(current_user: dict = Depends(get_current_user)):
    """
    Privacy / GDPR Right-to-Erasure:
    Permanently delete all scans and isolated media files for the current user.
    """
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    user_id = current_user["id"]
    cursor = db["analyses"].find({"user_id": user_id})

    count = 0
    for scan in cursor:
        file_path = scan.get("storage_path")
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception:
                pass
        count += 1

    db["analyses"].delete_many({"user_id": user_id})

    log_audit_event(
        action="DATA_ERASED",
        user_id=user_id,
        username=current_user["username"],
        details=f"User requested complete erasure of {count} scan records.",
    )

    return {
        "success": True,
        "deleted_count": count,
        "message": f"Successfully purged {count} scan records and associated storage.",
    }
