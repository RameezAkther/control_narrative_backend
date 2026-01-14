from datetime import datetime
from bson.objectid import ObjectId

from db.database import db

documents_collection = db["documents"]

def save_document(user_id: str, filename: str, filepath: str):
    doc = {
        "user_id": user_id,
        "filename": filename,
        "filepath": filepath,
        "uploaded_at": datetime.utcnow()
    }
    return documents_collection.insert_one(doc)

def get_documents_by_user(user_id: str, skip: int, limit: int):
    return list(
        documents_collection
        .find({"user_id": user_id})
        .skip(skip)
        .limit(limit)
        .sort("uploaded_at", -1)
    )

def delete_document(user_id: str, doc_id: str):
    return documents_collection.delete_one({
        "_id": ObjectId(doc_id),
        "user_id": user_id
    })

def get_document(user_id: str, doc_id: str):
    return documents_collection.find_one({
        "_id": ObjectId(doc_id),
        "user_id": user_id
    })

def count_user_documents(user_id: str):
    return documents_collection.count_documents({"user_id": user_id})
