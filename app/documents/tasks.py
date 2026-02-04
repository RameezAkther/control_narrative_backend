from bson import ObjectId
from pathlib import Path
import os

from app.agents.document_agent import DocumentAgent
from app.agents import image_parser_agent
from app.agents import pipeline
from app.documents.parsed_crud import (
    update_parsed_document,
    update_parsed_embeddings_info,
    update_progress
)
from app.utils.embeddings import embed_and_persist_document

from db.database import parsed_documents_collection as parsed_collection

def mock_image_parser(md_file_path, output_dir, new_filename):
    md_file_path = Path(md_file_path)
    output_dir = Path(output_dir)

    # Check source file
    if not md_file_path.exists() or md_file_path.suffix != ".md":
        raise FileNotFoundError("Source markdown file not found or invalid")
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read content
    content = md_file_path.read_text(encoding="utf-8")

    # Write to new markdown file
    output_file = output_dir / new_filename
    output_file.write_text(content, encoding="utf-8")

    print(f"Copied markdown to: {output_file.resolve()}")

def process_document_pipeline(document_id: str, file_path: str, parsing_strategy: str = "fast", parsed_folder: str = None):
    """
    Process a document through parsing, image enrichment, embedding, and agent pipeline.
    Args:
        document_id: ID of the document being processed
        file_path: Path to the document file
        parsing_strategy: 'fast' or 'accurate' parsing strategy
        parsed_folder: Optional folder to store parsed outputs
    """

    documentAgent = DocumentAgent()

    try:
        # Step: Document Parsing
        update_progress(
            document_id,
            step="document_parsing",
            message="Standardizing document structure"
        )

        parsed_output = documentAgent.run(file_path, parsing_strategy=parsing_strategy, output_dir=parsed_folder)

        if parsed_folder:
            parsed_output.setdefault("metadata", {})
            parsed_output["metadata"]["parsed_folder"] = parsed_folder
        update_parsed_document(document_id, "document_agent_output_file_path",parsed_output["metadata"]["markdown_path"])

        update_progress(
            document_id,
            step="document_parsing",
            message="Parsing completed"
        )

        # Step: Image Parsing & Enrichment
        update_progress(
            document_id,
            step="image_parsing",
            message="Enriching document with image descriptions"
        )

        print("Running Image Parser Agent...")
        image_parser_agent.run_image_parser(parsed_output["metadata"]["markdown_path"], parser_type= 2 if parsing_strategy == "accurate" else 1)
        # mock_image_parser(
        #     md_file_path=parsed_output["metadata"]["markdown_path"],
        #     output_dir=os.path.dirname(parsed_output["metadata"]["markdown_path"]),
        #     new_filename=os.path.basename(parsed_output["metadata"]["markdown_path"]).replace(".md", "_enriched.md")
        # )
        update_parsed_document(document_id, "document_image_parsed_output_file_path", parsed_output["metadata"]["markdown_path"]+"_enriched.md")

        update_progress(
            document_id,
            step="image_parsing",
            message="Image enrichment completed"
        )

        # Step: Embeddings Creation
        try:
            sections = parsed_output.get("sections")

            # prepare per-document embeddings folder inside parsed_folder
            persist_dir = os.path.join(parsed_folder, "embeddings") if parsed_folder else os.path.abspath("./db/chroma")
            if persist_dir and not os.path.exists(persist_dir):
                os.makedirs(persist_dir, exist_ok=True)

            update_progress(
                document_id,
                step="embeddings_started",
                message="Starting embedding creation"
            )

            emb_info = embed_and_persist_document(
                document_id=document_id,
                parsed_folder=parsed_folder,
                sections=sections,
                persist_directory=persist_dir,
            )

            # Persist info about embeddings
            update_parsed_embeddings_info(document_id, emb_info.get("collection_name"), emb_info.get("persist_directory"))

            update_progress(
                document_id,
                step="embeddings_created",
                message=f"Embeddings stored: {emb_info.get('collection_name')}"
            )
        except Exception as e:
            # Embedding failures should not crash the whole pipeline – record and continue
            print(f"Embeddings step failed: {e}")
            update_progress(
                document_id,
                step="embeddings_failed",
                message=str(e)
            )
        
        # Step: Run Agent Pipeline
        update_progress(
            document_id,
            step="agent_pipeline",
            message="Running document agent pipeline"
        )

        print("Running Agent Pipeline...")
        for fname in os.listdir(parsed_folder):
                if fname.lower().endswith('.md'):
                    enriched_md_path = os.path.join(parsed_folder, fname)
                    break
        pipeline.run_agent_pipeline(enriched_md_path, document_id=document_id)

        update_progress(
            document_id,
            step="completed",
            message="Document processing completed"
        )

    except Exception as e:
        print(f"Document processing failed: {e}")
        parsed_collection.update_one(
            {"document_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": "failed",
                    "progress": {
                        "step": "failed",
                        "message": str(e)
                    }
                }
            }
        )
