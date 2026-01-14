from datetime import datetime
from bson import ObjectId

from db.database import db

parsed_collection = db["parsed_documents"]

def create_parsed_document(document_id, user_id, parsing_strategy: str = "fast", parsed_folder: str = None):
    doc = {
        "document_id": ObjectId(document_id),
        "user_id": user_id,
        # Initial status: user just uploaded the file
        "status": "uploading the document",

        # Reserve where parsed outputs and settings live
        "document_agent": None,
        "control_logic_agent": None,
        "validator_agent": None,
        "code_generator_agent": None,

        # Save requested parsing strategy for transparency
        "parsing_strategy": parsing_strategy,

        # Where outputs (images, markdown) will be written
        "parsed_folder": parsed_folder,

        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    return parsed_collection.insert_one(doc)

def update_document_agent_output(document_id, output):
    # Update the parsed content; do not mark the overall pipeline as completed here —
    # further steps (embeddings, understanding, etc.) follow.
    return parsed_collection.update_one(
        {"document_id": ObjectId(document_id)},
        {
            "$set": {
                "document_agent": output,
                "updated_at": datetime.utcnow()
            }
        }
    )

def update_parsed_embeddings_info(document_id, chroma_collection: str, persist_directory: str):
    return parsed_collection.update_one(
        {"document_id": ObjectId(document_id)},
        {
            "$set": {
                "embeddings": {
                    "collection": chroma_collection,
                    "persist_directory": persist_directory
                },
                "updated_at": datetime.utcnow()
            }
        }
    )

def get_parsed_by_document_id(document_id):
    return parsed_collection.find_one(
        {"document_id": ObjectId(document_id)}
    )

def delete_parsed_by_document_id(document_id):
    """Delete parsed_documents record for a document_id.
    Returns the DeleteResult from pymongo so caller can inspect if a deletion happened.
    """
    return parsed_collection.delete_one({"document_id": ObjectId(document_id)})

def update_progress(document_id, step, message):
    """Update progress and normalize high-level `status` to one of the canonical stages.

    Mapping:
      - document_parsing -> parsing the document
      - embeddings_created -> embedding the parsed document
      - understanding_agent_pending -> understanding the document
      - control_logic_pending -> extracting control logic
      - validator_agent_pending -> validating the control logic
      - code_generator_pending -> generating code
      - loop_mapper_pending -> mapping loops
      - completed -> ready / completed
      - Any '*_failed' step sets status to 'failed'
    """

    step_to_status = {
        "document_parsing": "parsing the document",
        "embeddings_started": "embedding the parsed document",
        "embeddings_created": "embedding the parsed document",
        "understanding_agent_pending": "understanding the document",
        "control_logic_pending": "extracting control logic",
        "validator_agent_pending": "validating the control logic",
        "code_generator_pending": "generating code",
        "loop_mapper_pending": "mapping loops",
        "completed": "ready / completed",
        "uploading": "uploading the document",
    }

    status = None
    if isinstance(step, str) and step.endswith("_failed"):
        status = "failed"
    else:
        status = step_to_status.get(step)

    update_fields = {
        "progress": {"step": step, "message": message},
        "updated_at": datetime.utcnow()
    }

    if status:
        update_fields["status"] = status

    return parsed_collection.update_one(
        {"document_id": ObjectId(document_id)},
        {"$set": update_fields}
    )
