from fastapi import APIRouter, Depends
from app.utils.dependencies import get_current_user

router = APIRouter(prefix="/user", tags=["User"])

@router.get("/me")
def get_profile(current_user=Depends(get_current_user)):
    """
    Return basic user profile info.
    """
    user_data = {
        "name": current_user["name"],
        "email": current_user["email"],
        "documentsUploaded": current_user.get("documentsUploaded", 0),
    }
    return user_data
