import os
import json
import shutil

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.utils.dependencies import get_current_user
from app.documents.tasks import process_document_pipeline
from app.documents.crud import (
    save_document,
    get_documents_by_user,
    delete_document,
    get_document,
    count_user_documents
)
from app.documents.parsed_crud import (
    create_parsed_document,
    get_parsed_by_document_id,
    delete_parsed_by_document_id,
)

router = APIRouter(prefix="/documents", tags=["Documents"])

BASE_DIR = "./app/documents/user_documents"

@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    request: Request,
    files: list[UploadFile] = File(...),
    parsing_strategy: str = Form("fast"),
    current_user=Depends(get_current_user)
):
    print("=== UPLOAD DEBUG START ===")
    form = await request.form()
    print("Form keys:", list(form.keys()))
    user_id = str(current_user["_id"])
    user_folder = f"{BASE_DIR}/{user_id}"
    os.makedirs(user_folder, exist_ok=True)

    print(f"Uploading {len(files)} files for user {user_id} with parsing strategy '{parsing_strategy}'")

    uploaded = []

    for file in files:
        file_path = f"{user_folder}/{file.filename}"

        # STEP 1: Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"Saved file to {file_path}")
        doc_result = save_document(user_id, file.filename, file_path)
        document_id = str(doc_result.inserted_id)

        # create per-document parsed output folder inside the user's folder
        parsed_folder = os.path.abspath(f"{user_folder}/{file.filename}_parsed")
        os.makedirs(parsed_folder, exist_ok=True)

        # record requested parsing strategy and parsed folder
        create_parsed_document(document_id, user_id, parsing_strategy=parsing_strategy, parsed_folder=parsed_folder)

        # STEP 2: Start background processing
        background_tasks.add_task(
            process_document_pipeline,
            document_id,
            file_path,
            parsing_strategy,
            parsed_folder
        )
        
        uploaded.append({
            "document_id": document_id,
            "filename": file.filename,
            "parsing_strategy": parsing_strategy,
            "parsed_folder": parsed_folder
        })

    return {
        "message": "Upload successful. Processing started.",
        "documents": uploaded
    }

@router.get("/list")
def list_user_documents(
    page: int = 1,
    page_size: int = 10,
    current_user=Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    skip = (page - 1) * page_size
    total_docs = count_user_documents(user_id)
    docs = get_documents_by_user(user_id, skip, page_size)

    # Convert ObjectId to string
    for d in docs:
        d["_id"] = str(d["_id"])

    total_pages = (total_docs + page_size - 1) // page_size

    return {
        "documents": docs,
        "page": page,
        "page_size": page_size,
        "total_documents": total_docs,
        "total_pages": total_pages
    }

@router.delete("/{doc_id}")
def delete_user_document(
    doc_id: str,
    current_user=Depends(get_current_user)
):
    user_id = str(current_user["_id"])
    document = get_document(user_id, doc_id)
    if not document:
        raise HTTPException(404, "Document not found or not owned by user")

    # delete file
    if os.path.exists(document["filepath"]):
        try:
            os.remove(document["filepath"])
        except Exception as e:
            print(f"Error removing file: {e}")

    # Delete parsed document DB entry and files
    parsed = get_parsed_by_document_id(doc_id)
    if parsed and parsed.get("user_id") == user_id:
        parsed_folder = parsed.get("parsed_folder")
        if parsed_folder and os.path.exists(parsed_folder):
            try:
                shutil.rmtree(parsed_folder)
                print(f"Deleted parsed folder: {parsed_folder}")
            except Exception as e:
                print(f"Failed to delete parsed folder {parsed_folder}: {e}")
        delete_parsed_by_document_id(doc_id)
    else:
        # Fallback deletion strategy
        guessed_parsed = os.path.abspath(f"{BASE_DIR}/{user_id}/{document.get('filename')}_parsed")
        if os.path.exists(guessed_parsed):
            try:
                shutil.rmtree(guessed_parsed)
            except Exception as e:
                print(f"Failed to delete guessed folder: {e}")

    delete_document(user_id, doc_id)
    return {"message": "Document and parsed data deleted successfully"}

@router.get("/status/{document_id}")
def get_document_status(
    document_id: str,
    current_user=Depends(get_current_user)
):
    parsed = get_parsed_by_document_id(document_id)
    
    # Fetch original document for filename
    doc_meta = get_document(str(current_user["_id"]), document_id)
    filename = doc_meta["filename"] if doc_meta else "Unknown Document"

    if not parsed or parsed["user_id"] != str(current_user["_id"]):
        raise HTTPException(404, "Document not found")

    # STRICT ORDER based on your update_progress calls
    # This determines the percentage calculation
    STATUS_ORDER = [
        "queued", # Default initial state
        "document_parsing",
        "image_parsing",
        "embeddings_started",
        "embeddings_created",
        "agent_pipeline",
        "understanding_agent_pending",
        "understanding_agent_completed",
        "control_logic_pending",
        "control_logic_completed",
        "loop_mapper_pending",
        "loop_mapper_completed",
        "validator_agent_pending",
        "validator_agent_completed",
        "code_generator_completed",
        "mindmap_generator_agent_pending",
        "completed" # Final step
    ]

    # Get the status key (step) and the specific message
    # We prioritize the specific message saved in 'progress.message'
    
    # FIX: Use the granular 'step' from progress to determine the stage, 
    # as 'status' field is often too high-level (e.g. "understanding the document")
    progress_data = parsed.get("progress", {})
    status_key = progress_data.get("step", "queued")
    
    # Should use the human readable status for the returned "status" field or logic?
    # The frontend uses this key for display text sometimes, but mostly uses stage/message.
    # The frontend expects "status" to be valid-ish. 
    # Let's return the high-level status as "status" but use "step" for stage calc.
    high_level_status = parsed.get("status", "queued")
    
    # Calculate Stage Index
    try:
        # +1 because we want 1-based indexing for the UI
        stage = STATUS_ORDER.index(status_key) + 1
    except ValueError:
        # If status isn't in the list (e.g. error or unknown step), default to 0 or max
        # Check if actual status is failed
        if "fail" in high_level_status.lower() or "fail" in status_key.lower():
             stage = len(STATUS_ORDER) # Or handle as error state
        else:
             stage = 0 

    print(f"Document {document_id} step: {status_key} -> Stage {stage}/{len(STATUS_ORDER)}")

    return {
        "status": high_level_status, # Keep returning the high-level status string
        "stage": stage,
        "total_stages": len(STATUS_ORDER), # Send total to frontend for accurate math
        "progress": progress_data, # Contains the specific 'message' from update_progress
        "filename": filename
    }

@router.get("/{document_id}/artifact/{file_key}")
def get_document_artifact(
    document_id: str,
    file_key: str,
    current_user=Depends(get_current_user)
):
    """
    Fetches the specific generated artifact (JSON, Markdown, or ST code)
    based on the file_key requested by the frontend.
    """
    user_id = str(current_user["_id"])
    
    # 1. Verify ownership and existence
    parsed_record = get_parsed_by_document_id(document_id)
    if not parsed_record or parsed_record["user_id"] != user_id:
        raise HTTPException(404, "Document not found")
        
    parsed_folder = parsed_record.get("parsed_folder")
    if not parsed_folder or not os.path.exists(parsed_folder):
        raise HTTPException(404, "Parsed data directory not found")

    # 2. Get the original filename to locate the Markdown file (which uses the filename)
    doc_meta = get_document(user_id, document_id)
    if not doc_meta:
        raise HTTPException(404, "Original document metadata missing")
        
    original_filename = doc_meta["filename"]
    # Remove extension for the markdown mapping (e.g. MyDoc.pdf -> MyDoc)
    filename_stem = os.path.splitext(original_filename)[0]

    # 3. Map frontend keys to backend file paths
    # Paths are relative to the `parsed_folder`
    FILE_MAPPING = {
        # 'parsed' refers to the markdown file inside the images/ folder
        "parsed": f"{filename_stem}.md",
        "markdown": f"{filename_stem}.md",
        
        # Agent outputs inside agent_generated_files/
        "summary": os.path.join("agent_generated_files", "1_summary.json"),
        "logic": os.path.join("agent_generated_files", "2_logic_extracted.json"),
        "loop_map": os.path.join("agent_generated_files", "3_loop_map.json"),
        "validation": os.path.join("agent_generated_files", "4_validation.json"),
        "plc_code": os.path.join("agent_generated_files", "5_plc_code.st"),
        "mindmaps_index": os.path.join("agent_generated_files", "6_mindmaps_index.json"),
    }
    
    # Dynamic handling for individual mindmap files (e.g., "mindmap_Loop-101")
    if file_key.startswith("mindmap_") and file_key != "mindmaps_index":
        # Extract loop name from key (e.g., mindmap_P-101 -> P-101)
        loop_name_requested = file_key.replace("mindmap_", "")
        FILE_MAPPING[file_key] = os.path.join("agent_generated_files", "mindmaps", f"{loop_name_requested}.json")

    print(f"Requested artifact key: {file_key}")

    if file_key not in FILE_MAPPING:
        raise HTTPException(400, f"Invalid artifact type: {file_key}")
    print(f"Fetching artifact '{file_key}' for document '{document_id}'")
    target_relative_path = FILE_MAPPING[file_key]
    full_path = os.path.join(parsed_folder, target_relative_path)
    print(target_relative_path)
    print(f"Resolved file path: {full_path}")
    if not os.path.exists(full_path):
        # Specific error message for frontend
        raise HTTPException(404, f"File not found: {target_relative_path}. The agent might still be processing this step.")

    # 4. Read and Return Content
    try:
        # Determine if we should return JSON object or Plain Text
        is_json = file_key in ["summary", "logic_extracted", "loop_map", "validation", "mindmaps_index"] or file_key.startswith("mindmap_")
        
        with open(full_path, "r", encoding="utf-8") as f:
            if is_json:
                content = json.load(f)
                return content  # FastAPI automatically converts dict/list to JSONResponse
            else:
                content = f.read()
                return PlainTextResponse(content)
                
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=500, 
            content={"error": "File contains invalid JSON", "path": target_relative_path}
        )
    except Exception as e:
        print(f"Error reading artifact {full_path}: {e}")
        raise HTTPException(500, f"Failed to read file: {str(e)}")