import logging
from typing import Dict, Any, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, ConfigurationError

from config import settings

logger = logging.getLogger("deepguard.database")

_mongo_client: Optional[MongoClient] = None
_mongo_db: Optional[Database] = None


def get_client() -> Optional[MongoClient]:
    """Return the active MongoClient instance if initialized."""
    global _mongo_client
    return _mongo_client


def get_database() -> Optional[Database]:
    """Return the active MongoDB database instance if connected."""
    global _mongo_db
    return _mongo_db


def check_database_connection() -> Dict[str, Any]:
    """
    Test and validate MongoDB Atlas connection without modifying data.
    Never exposes passwords or private credentials.
    """
    global _mongo_client, _mongo_db

    if not settings.is_mongodb_configured:
        return {
            "status": "not_configured",
            "message": "MongoDB Atlas URI is not configured. Add MONGODB_URI to your .env file.",
            "database_name": settings.mongodb_db_name,
            "connected": False,
        }

    try:
        # Re-use existing client or create temporary test client if needed
        client = _mongo_client
        if client is None:
            client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
                connectTimeoutMS=settings.mongodb_timeout_ms,
                socketTimeoutMS=settings.mongodb_timeout_ms,
                appname="DeepGuard",
            )

        # Ping admin database to verify real round-trip connectivity
        client.admin.command("ping")

        return {
            "status": "connected",
            "message": "Connected to MongoDB successfully.",
            "database_name": settings.mongodb_db_name,
            "connected": True,
        }

    except ServerSelectionTimeoutError:
        return {
            "status": "disconnected",
            "message": "Connection to MongoDB Atlas timed out. Check network access, IP whitelist, or cluster status.",
            "database_name": settings.mongodb_db_name,
            "connected": False,
        }
    except ConfigurationError as err:
        return {
            "status": "disconnected",
            "message": f"MongoDB configuration error: {err}",
            "database_name": settings.mongodb_db_name,
            "connected": False,
        }
    except PyMongoError as err:
        return {
            "status": "disconnected",
            "message": f"MongoDB database error: {type(err).__name__}",
            "database_name": settings.mongodb_db_name,
            "connected": False,
        }
    except Exception as err:
        return {
            "status": "disconnected",
            "message": f"Unexpected database connection error: {type(err).__name__}",
            "database_name": settings.mongodb_db_name,
            "connected": False,
        }


def init_db() -> Dict[str, Any]:
    """
    Initialize global MongoDB connection on application startup.
    Fails gracefully if Atlas is not reachable.
    """
    global _mongo_client, _mongo_db

    if not settings.is_mongodb_configured:
        logger.info("MongoDB URI not configured. Starting in local unauthenticated mode.")
        return check_database_connection()

    try:
        _mongo_client = MongoClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
            connectTimeoutMS=settings.mongodb_timeout_ms,
            appname="DeepGuard",
        )
        # Verify connection
        _mongo_client.admin.command("ping")
        _mongo_db = _mongo_client[settings.mongodb_db_name]
        logger.info("MongoDB Atlas connected successfully: %s", settings.mongodb_db_name)
        return {
            "status": "connected",
            "message": "Connected to MongoDB successfully.",
            "database_name": settings.mongodb_db_name,
            "connected": True,
        }
    except Exception as exc:
        logger.warning("MongoDB Atlas initialization warning: %s", exc)
        _mongo_client = None
        _mongo_db = None
        return check_database_connection()


def close_db():
    """Close MongoDB client connection on shutdown."""
    global _mongo_client, _mongo_db
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB connection closed.")
        except Exception:
            pass
        finally:
            _mongo_client = None
            _mongo_db = None
