from db.database import db

users_collection = db["users"]

def get_user_by_email(email: str):
    return users_collection.find_one({"email": email})

def create_user(user_data: dict):
    return users_collection.insert_one(user_data)

def update_user_password(email: str, hashed_password: str):
    return users_collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}}
    )