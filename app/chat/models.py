from pydantic import BaseModel, Field, BeforeValidator
from typing import Optional, List, Annotated
from datetime import datetime

# Helper to convert MongoDB ObjectId to string for JSON responses
PyObjectId = Annotated[str, BeforeValidator(str)]

class ChatSessionModel(BaseModel):
    """
    Represents a conversation thread.
    Points to User and Documents.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId = Field(...)  # Points to users collection
    
    name: str = "New Chat"
    mode: str = "RAG"
    
    # List of document IDs (ObjectIds) this chat has access to
    document_ids: List[PyObjectId] = []
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ChatMessageModel(BaseModel):
    """
    Represents a single message.
    Points to ChatSession and User.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    session_id: PyObjectId = Field(...) # Points to chat_sessions collection
    user_id: PyObjectId = Field(...)    # Points to users collection
    
    role: str  # "user" or "assistant"
    content: str
    
    # Context Logic: Which previous messages were active when this was sent?
    active_context_ids: Optional[List[str]] = []
    
    # RAG Logic: Which chunks/citations were used?
    citations: Optional[List[dict]] = []
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True