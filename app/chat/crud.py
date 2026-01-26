from bson import ObjectId
from datetime import datetime
from pymongo import MongoClient

from app.chat.models import ChatSessionModel, ChatMessageModel

client = MongoClient("mongodb://localhost:27017/")
db = client["control_narrative_ai"]
chat_sessions_collection = db["chat_sessions"]
chat_messages_collection = db["chat_messages"]

# --- Session Operations ---

def create_chat_session(user_id: str, data: dict) -> str:
    """Creates a new session."""
    session = ChatSessionModel(
        user_id=user_id,
        name=data.get("name", "New Chat"),
        mode=data.get("mode", "RAG"),
        document_ids=data.get("document_ids", [])
    )
    result = chat_sessions_collection.insert_one(
        session.model_dump(by_alias=True, exclude=["id"])
    )
    return str(result.inserted_id)

def get_user_chat_sessions(user_id: str):
    """List all chats for sidebar."""
    sessions = chat_sessions_collection.find({"user_id": user_id}).sort("updated_at", -1)
    return [ChatSessionModel(**s) for s in sessions]

def get_chat_session(session_id: str, user_id: str):
    """Get single session details."""
    try:
        oid = ObjectId(session_id)
    except:
        return None
        
    session = chat_sessions_collection.find_one({"_id": oid, "user_id": user_id})
    return ChatSessionModel(**session) if session else None

def update_chat_session(session_id: str, user_id: str, new_name: str):
    """Renames the chat session."""
    chat_sessions_collection.update_one(
        {"_id": ObjectId(session_id), "user_id": user_id},
        {"$set": {"name": new_name, "updated_at": datetime.utcnow()}}
    )

def delete_chat_session(session_id: str, user_id: str):
    """Deletes the session AND all its messages."""
    # 1. Delete Messages first
    chat_messages_collection.delete_many({"session_id": session_id, "user_id": user_id})
    # 2. Delete Session
    chat_sessions_collection.delete_one({"_id": ObjectId(session_id), "user_id": user_id})

# --- Message Operations ---

def add_message(
    session_id: str, 
    user_id: str, 
    role: str, 
    content: str, 
    context_ids: list = [], 
    document_ids: list = [],
    citations: list = [],
    mode: str = "RAG",
    artifacts: list = [],
    summary: str = ""
):
    msg = ChatMessageModel(
        session_id=session_id,
        user_id=user_id,
        role=role,
        content=content,
        active_context_ids=context_ids,
        citations=citations,
        document_ids=document_ids, # (Optional: ensure your model has this field if you want to store it)
        mode=mode,             # <--- ADD THIS
        artifacts=artifacts,    # <--- ADD THIS
        summary=summary         # <--- ADD THIS
    )
    
    result = chat_messages_collection.insert_one(
        msg.model_dump(by_alias=True, exclude=["id"])
    )
    
    # Update the session's 'updated_at' timestamp
    chat_sessions_collection.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"updated_at": datetime.utcnow()}}
    )
    
    # Return the full object so we can send it back to frontend
    msg.id = str(result.inserted_id)
    return msg

def get_session_messages(session_id: str, user_id: str):
    """Get history."""
    msgs = chat_messages_collection.find(
        {"session_id": session_id, "user_id": user_id}
    ).sort("created_at", 1)
    return [ChatMessageModel(**m) for m in msgs]
