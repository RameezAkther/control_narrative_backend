from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.auth.schemas import RegisterSchema, LoginSchema, TokenResponse
from app.auth.password_handler import hash_password, verify_password
from app.auth.jwt_handler import create_access_token
from app.users.crud import get_user_by_email, create_user, update_user_password
from app.utils.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=TokenResponse)
def register(payload: RegisterSchema):
    if get_user_by_email(payload.email):
        raise HTTPException(status_code=400, detail="User already exists")

    new_user = {
        "name": payload.name,
        "email": payload.email,
        "password": hash_password(payload.password)
    }

    create_user(new_user)

    token = create_access_token({"email": payload.email})
    return TokenResponse(access_token=token)

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginSchema):
    user = get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(payload.password, user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token({"email": payload.email})
    return TokenResponse(access_token=token)

class ChangePasswordSchema(BaseModel):
    currentPassword: str
    newPassword: str

@router.post("/change-password")
def change_password(payload: ChangePasswordSchema, current_user=Depends(get_current_user)):
    user = current_user

    # Verify current password
    if not verify_password(payload.currentPassword, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect current password")

    # Validate new password length
    if len(payload.newPassword) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters long")

    # Hash the new password
    new_hashed = hash_password(payload.newPassword)

    # Update user in DB
    update_user_password(user["email"], new_hashed)

    return {"message": "Password updated successfully!"}
