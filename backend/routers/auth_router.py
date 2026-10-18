import datetime
from typing import Optional
from bson import ObjectId
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr, Field

from auth import hash_password, verify_password, create_access_token, get_current_user
from database import get_database

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=6, max_length=128)
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    identifier: str = Field(..., description="Email or Username")
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    full_name: Optional[str] = None
    created_at: Optional[datetime.datetime] = None


class AuthResponse(BaseModel):
    user: UserResponse
    token: str
    token_type: str = "bearer"


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """Register a new user account."""
    db = get_database()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is offline. Please check MongoDB configuration.",
        )

    # Check for existing user with same username or email
    clean_username = req.username.strip().lower()
    clean_email = req.email.strip().lower()

    existing = db["users"].find_one({
        "$or": [
            {"username": clean_username},
            {"email": clean_email},
        ]
    })
    if existing:
        if existing.get("username") == clean_username:
            raise HTTPException(status_code=400, detail="Username is already taken.")
        raise HTTPException(status_code=400, detail="Email is already registered.")

    now = datetime.datetime.now(datetime.timezone.utc)
    user_doc = {
        "username": clean_username,
        "email": clean_email,
        "password_hash": hash_password(req.password),
        "full_name": req.full_name.strip() if req.full_name else clean_username,
        "role": "user",
        "created_at": now,
        "updated_at": now,
    }

    result = db["users"].insert_one(user_doc)
    user_id = str(result.inserted_id)

    token = create_access_token(user_id=user_id, username=clean_username, role="user")

    # Record audit log
    try:
        db["audit_logs"].insert_one({
            "action": "USER_REGISTER",
            "user_id": user_id,
            "username": clean_username,
            "timestamp": now,
            "details": f"User account registered for {clean_username}",
        })
    except Exception:
        pass

    return AuthResponse(
        user=UserResponse(
            id=user_id,
            username=clean_username,
            email=clean_email,
            role="user",
            full_name=user_doc["full_name"],
            created_at=now,
        ),
        token=token,
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    """Authenticate with username or email and password."""
    db = get_database()
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is offline. Please check MongoDB configuration.",
        )

    clean_id = req.identifier.strip().lower()

    user = db["users"].find_one({
        "$or": [
            {"username": clean_id},
            {"email": clean_id},
        ]
    })
    if not user or not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials. Please verify your username/email and password.",
        )

    user_id = str(user["_id"])
    role = user.get("role", "user")
    token = create_access_token(user_id=user_id, username=user["username"], role=role)

    # Record login in audit log
    try:
        db["audit_logs"].insert_one({
            "action": "USER_LOGIN",
            "user_id": user_id,
            "username": user["username"],
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "details": "User logged in successfully",
        })
    except Exception:
        pass

    return AuthResponse(
        user=UserResponse(
            id=user_id,
            username=user["username"],
            email=user["email"],
            role=role,
            full_name=user.get("full_name"),
            created_at=user.get("created_at"),
        ),
        token=token,
    )


@router.get("/me", response_model=UserResponse)
def get_profile(current_user: dict = Depends(get_current_user)):
    """Retrieve authenticated user's profile."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        role=current_user.get("role", "user"),
        full_name=current_user.get("full_name"),
        created_at=current_user.get("created_at"),
    )


@router.post("/logout")
def logout(current_user: dict = Depends(get_current_user)):
    """Acknowledge user logout."""
    db = get_database()
    if db is not None:
        try:
            db["audit_logs"].insert_one({
                "action": "USER_LOGOUT",
                "user_id": current_user["id"],
                "username": current_user["username"],
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "details": "User logged out",
            })
        except Exception:
            pass
    return {"message": "Logged out successfully."}
