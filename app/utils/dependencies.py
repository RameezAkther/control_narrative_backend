from fastapi import Header, HTTPException

from app.auth.jwt_handler import decode_token
from app.users.crud import get_user_by_email

def get_current_user(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(401, "Missing authorization header")

    token = authorization.split(" ")[1]
    decoded = decode_token(token)

    if decoded is None:
        raise HTTPException(401, "Invalid or expired token")

    user = get_user_by_email(decoded["email"])
    if user is None:
        raise HTTPException(401, "User not found")

    return user
