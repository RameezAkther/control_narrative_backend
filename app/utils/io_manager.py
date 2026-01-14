# utils.py
import os
import ast
import json
from pathlib import Path

def create_output_dir(source_file_path: str) -> Path:
    """
    Creates an 'agent_generated_files' folder inside the same directory 
    as the source markdown file.
    
    E.g., "C:/Projects/Docs/MyDoc.md" -> "C:/Projects/Docs/agent_generated_files"
    """
    source_path = Path(source_file_path).resolve()
    
    # Get the parent directory of the source file
    parent_dir = source_path.parent
    
    # Define the target directory inside that parent
    target_dir = parent_dir / "agent_generated_files"
    
    # Create the directory
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"📂 Output directory ready: {target_dir}")
    return target_dir

def save_json(data: dict, output_dir: Path, filename: str):
    """Saves a dictionary as a pretty-printed JSON file."""
    file_path = output_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved JSON: {file_path}")

def save_text(text: str, output_dir: Path, filename: str):
    """Saves raw text (like code or reports) to a file."""
    file_path = output_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"💾 Saved Text: {file_path}")

def extract_json_from_result(crew_result):
    """
    Safe extraction of JSON from CrewAI result object.
    Handles Dicts, CrewOutput objects, JSON strings, and Python Dict strings.
    """
    # 1. If it's already a Dict, return it immediately!
    # (This prevents the error you are seeing)
    if isinstance(crew_result, dict):
        return crew_result

    # 2. Try to get direct JSON dict (Newer CrewAI)
    if hasattr(crew_result, 'json_dict') and crew_result.json_dict:
        return crew_result.json_dict
    
    # 3. Extract text from result object if needed
    raw_str = ""
    if hasattr(crew_result, 'raw'):
        raw_str = crew_result.raw
    else:
        raw_str = str(crew_result)

    # 4. Clean up Markdown
    # Removes ```json and ``` fences
    raw_str = raw_str.replace("```json", "").replace("```", "").strip()
            
    try:
        # Try standard JSON parsing
        return json.loads(raw_str)
    except json.JSONDecodeError:
        try:
            # 5. Fallback: Parse Python Dictionary String (Single Quotes)
            # This fixes "{'mappings': ...}"
            return ast.literal_eval(raw_str)
        except Exception:
            # 6. Final Fallback: Return wrapped error
            return {
                "raw_content": raw_str, 
                "error": "Could not parse strict JSON"
            }