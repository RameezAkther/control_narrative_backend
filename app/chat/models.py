from pydantic import BaseModel, Field, BeforeValidator
from typing import Optional, List, Annotated
from datetime import datetime

# Helper to convert MongoDB ObjectId to string for JSON responses
PyObjectId = Annotated[str, BeforeValidator(str)]

class ChatSessionModel(BaseModel):
    """
    Represents a conversation thread.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    user_id: PyObjectId = Field(...) 
    
    name: str = "New Chat"
    mode: str = "RAG"
    document_ids: List[PyObjectId] = []
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class ChatMessageModel(BaseModel):
    """
    Represents a single message.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    session_id: PyObjectId = Field(...) 
    user_id: PyObjectId = Field(...)    
    
    role: str  # "user" or "assistant"
    content: str
    
    # Context Logic
    active_context_ids: Optional[List[str]] = []
    
    # RAG Logic
    citations: Optional[List[dict]] = []
    
    # 3. ADD THESE FIELDS to store the selections in DB
    mode: str = "RAG"
    artifacts: Optional[List[str]] = []
    document_ids: Optional[List[str]] = []

    summary: str = ""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True