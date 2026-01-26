from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# --- Shared ---
class ChatMode(str):
    RAG = "RAG"
    REACT = "ReACT"
    FLARE = "Flare"

# --- Message Schemas ---
class MessageCreate(BaseModel):
    message: str
    mode: str = "RAG"
    document_ids: List[str] = []
    active_context_ids: List[str] = []
    
    # 1. ADD THIS FIELD (This was likely causing the 422 error)
    artifacts: List[str] = [] 

class MessageResponse(BaseModel):
    id: str = Field(alias="_id")
    role: str
    content: str
    citations: Optional[List[dict]] = []
    created_at: datetime
    
    # 2. Add these to return metadata to the UI
    mode: str = "RAG"
    artifacts: List[str] = []
    document_ids: List[str] = [] 

    class Config:
        populate_by_name = True
        from_attributes = True

# --- Chat Session Schemas ---
class ChatCreate(BaseModel):
    name: str = "New Chat"
    mode: str = "RAG"
    document_ids: List[str] = []

class ChatUpdate(BaseModel):
    name: str

class ChatSessionResponse(BaseModel):
    id: str = Field(alias="_id")
    name: str
    mode: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        populate_by_name = True
        from_attributes = True