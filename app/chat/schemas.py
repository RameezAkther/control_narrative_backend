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
    # The IDs of docs selected for THIS specific message
    document_ids: List[str] = []
    # The IDs of previous messages to use as context
    active_context_ids: List[str] = []

class MessageResponse(BaseModel):
    id: str = Field(alias="_id")
    role: str
    content: str
    # Add this field to return stored context
    citations: Optional[List[dict]] = [] 
    created_at: datetime
    
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

