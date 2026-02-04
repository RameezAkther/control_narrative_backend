"""Embedding utilities: create per-document Chroma collections and persist embeddings locally.

Uses sentence-transformers for local embeddings and ChromaDB for vector storage.
Each document will be stored in its own Chroma collection named `doc_<document_id>` so
retrieval can be scoped to a single document.
"""
from typing import List, Dict, Optional
import os

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    # Settings are no longer strictly required for basic usage in new versions
except Exception as e:
    raise RuntimeError("Missing dependencies for embeddings. Install 'sentence-transformers' and 'chromadb'.") from e

def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Simple sliding-window chunker on characters."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= L:
            break
        start = max(0, end - overlap)
    return chunks

def embed_and_persist_document(
    document_id: str,
    parsed_folder: str,
    sections: Optional[List[Dict]] = None,
    model_name: str = "all-mpnet-base-v2",
    persist_directory: str = "./db/chroma",
) -> Dict:
    """Embed a document and persist embeddings to a Chroma collection.

    Args:
        document_id: string id of the document (ObjectId string)
        parsed_folder: path where markdown/images live (used for metadata)
        sections: optional list of {title, content} dicts.
        model_name: sentence-transformers model name
        persist_directory: where Chroma will persist data

    Returns: info dict with collection name and counts
    """
    # 1) Ensure output directory exists
    # os.makedirs(persist_directory, exist_ok=True)

    # 2) Prepare texts
    texts = []
    metadatas = []
    ids = []

    if not sections:
        # Attempt to read markdown file (first *.md in folder)
        md_path = None
        if parsed_folder and os.path.exists(parsed_folder):
            for fname in os.listdir(parsed_folder):
                if fname.lower().endswith('_enriched.md'):
                    md_path = os.path.join(parsed_folder, fname)
                    break
        print(md_path)
        if not md_path or not os.path.exists(md_path):
             # Return empty if no content found, rather than crashing
            print(f"Warning: No markdown found in {parsed_folder}, skipping embeddings.")
            return {"collection_name": f"doc_{document_id}", "num_embeddings": 0}

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        sections = [{"title": os.path.basename(md_path), "content": content}]

    counter = 0
    for s in sections:
        title = s.get('title', 'section')
        content = s.get('content', '')
        # chunk content
        for chunk in _chunk_text(content):
            counter += 1
            texts.append(chunk)
            metadatas.append({
                "document_id": document_id,
                "section_title": title,
                "parsed_folder": parsed_folder
            })
            ids.append(f"{document_id}_{counter}")

    if not texts:
        return {"collection_name": f"doc_{document_id}", "num_embeddings": 0}

    # 3) Create embeddings
    print(f"Generating embeddings for {len(texts)} chunks...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    # 4) Persist to ChromaDB (Modern API)
    collection_name = f"doc_{document_id}"
    try:
        # UPDATED: Use PersistentClient directly. 
        # This automatically handles saving to disk (SQLite).
        client = chromadb.PersistentClient(path=persist_directory)
        
        collection_name = f"doc_{document_id}"
        
        # Get or create the collection
        collection = client.get_or_create_collection(name=collection_name)
        
        # Add data
        # Note: Chroma expects python lists for embeddings, not numpy arrays
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist()
        )
        
        # NOTE: client.persist() is NOT needed in new Chroma versions. 
        # Data is auto-persisted.
    
    except Exception as e:
        raise RuntimeError(f"ChromaDB error: {e}") from e

    return {
        "collection_name": collection_name, 
        "num_embeddings": len(ids), 
        "persist_directory": os.path.abspath(persist_directory)
    }