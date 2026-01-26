from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List
import time

# Adjust these imports to match your project structure
from app.utils.dependencies import get_current_user
from app.chat.rag_engine import rag_engine
from app.chat import schemas, crud

router = APIRouter(prefix="/chats", tags=["Chat"])

# --- Session Management ---

@router.post("/", response_model=schemas.ChatSessionResponse)
def create_chat(
    payload: schemas.ChatCreate,
    current_user = Depends(get_current_user)
):
    """Create a new chat session."""
    session_id = crud.create_chat_session(str(current_user["_id"]), payload.model_dump())
    new_session = crud.get_chat_session(session_id, str(current_user["_id"]))
    return new_session

@router.get("/", response_model=List[schemas.ChatSessionResponse])
def get_chats(current_user = Depends(get_current_user)):
    """List all chats for the sidebar."""
    return crud.get_user_chat_sessions(str(current_user["_id"]))

@router.get("/{chat_id}/history", response_model=List[schemas.MessageResponse])
def get_chat_history(
    chat_id: str,
    current_user = Depends(get_current_user)
):
    """Load messages when opening a chat."""
    session = crud.get_chat_session(chat_id, str(current_user["_id"]))
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    return crud.get_session_messages(chat_id, str(current_user["_id"]))

@router.patch("/{chat_id}")
def update_chat_name(
    chat_id: str,
    payload: schemas.ChatUpdate,
    current_user = Depends(get_current_user)
):
    """Rename a chat session."""
    session = crud.get_chat_session(chat_id, str(current_user["_id"]))
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    crud.update_chat_session(chat_id, str(current_user["_id"]), payload.name)
    return {"status": "success", "name": payload.name}

@router.delete("/{chat_id}")
def delete_chat(
    chat_id: str,
    current_user = Depends(get_current_user)
):
    """Delete a session and all its messages."""
    session = crud.get_chat_session(chat_id, str(current_user["_id"]))
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")
        
    crud.delete_chat_session(chat_id, str(current_user["_id"]))
    return {"status": "deleted"}

# --- Messaging (The Core Logic) ---

@router.post("/{chat_id}/message", response_model=schemas.MessageResponse)
async def send_message(
    chat_id: str,
    payload: schemas.MessageCreate,
    current_user = Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    
    # 1. Check Session
    session = crud.get_chat_session(chat_id, user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # 2. Save User Message
    crud.add_message(
        session_id=chat_id,
        user_id=user_id,
        role="user",
        content=payload.message,
        context_ids=payload.active_context_ids,
        document_ids=payload.document_ids,
        mode=payload.mode,
        artifacts=payload.artifacts
    )

    # 3. Generate Response
    ai_response_text = ""
    citations_data = []
    response_summary = ""  # Initialize

    try:
        # --- FIX: UNPACK 3 VALUES HERE ---
        response_tuple = await rag_engine.generate_response(
            query=payload.message,
            document_ids=payload.document_ids,
            active_context_ids=payload.active_context_ids,
            artifacts=payload.artifacts,
            mode=payload.mode
        )
        
        # Handle tuple unpacking safely
        if isinstance(response_tuple, tuple):
            if len(response_tuple) == 3:
                ai_response_text, citations_data, response_summary = response_tuple
            elif len(response_tuple) == 2:
                ai_response_text, citations_data = response_tuple
        else:
            ai_response_text = response_tuple
            
    except Exception as e:
        print(f"RAG Error: {e}")
        ai_response_text = "I encountered an error analyzing your documents."

    # 4. Save Assistant Message with Summary
    ai_message = crud.add_message(
        session_id=chat_id,
        user_id=user_id,
        role="assistant",
        content=ai_response_text,
        document_ids=payload.document_ids,
        citations=citations_data,
        mode=payload.mode,
        artifacts=payload.artifacts,
        summary=response_summary # <--- PASS THE SUMMARY TO DB
    )

    return ai_message