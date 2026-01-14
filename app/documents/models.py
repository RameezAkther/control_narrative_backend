from pydantic import BaseModel
from datetime import datetime

class DocumentModel(BaseModel):
    id: str
    user_id: str
    filename: str
    filepath: str
    uploaded_at: datetime
