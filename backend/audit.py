import datetime
import logging
from typing import Optional, Dict, Any
from database import get_database

logger = logging.getLogger("deepguard.audit")


def log_audit_event(
    action: str,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    details: Optional[str] = None,
    ip_address: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Record a security or operational event to the audit_logs collection.
    Never fails or throws unhandled exceptions that could block user operations.
    """
    db = get_database()
    if db is None:
        logger.info("[AUDIT-FALLBACK] %s by %s: %s", action, username or user_id or "anonymous", details)
        return False

    try:
        doc = {
            "action": action,
            "user_id": user_id,
            "username": username or (user_id if user_id else "anonymous"),
            "details": details or "",
            "ip_address": ip_address,
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
        }
        db["audit_logs"].insert_one(doc)
        return True
    except Exception as exc:
        logger.warning("Failed to record audit log: %s", exc)
        return False
