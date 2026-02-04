import os
from app.agents.understanding_agent import UnderstandingAgentRunner, parse_markdown_file
from app.agents.control_logic_agent import LogicAgentRunner
from app.agents.mapper_agent import LoopMapperRunner
from app.agents.validator_agent import ValidatorRunner
from app.agents.code_generator_agent import CodeGeneratorRunner

from app.utils.io_manager import create_output_dir, save_json, save_text, extract_json_from_result
from app.documents.parsed_crud import update_progress
from app.documents.parsed_crud import update_parsed_document

def run_agent_pipeline(md_file_path: str, document_id: str = None):
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key: ")

    print(f"\nSTARTING PIPELINE FOR: {md_file_path}")
    
    # Create the structured output folder
    output_dir = create_output_dir(md_file_path)

    # Parse the Markdown Document
    try:
        sections_data = parse_markdown_file(md_file_path)
        print(f"Document Parsed: {len(sections_data)} sections found.")
    except Exception as e:
        print(f"Error parsing file: {e}")
        return
    
    update_progress(
        document_id,
        step="understanding_agent_pending",
        message="Starting understanding agent"
    )

    # --- 2. Understanding Agent ---
    print("\nSTEP 1/5: Running Understanding Agent...")
    understanding_runner = UnderstandingAgentRunner()
    summary_result = understanding_runner.run(sections_data)
    
    # Extract & Save
    summary_json = extract_json_from_result(summary_result)
    save_json(summary_json, output_dir, "1_summary.json")
    update_parsed_document(document_id, "understanding_agent_output_file_path", os.path.join(output_dir, "1_summary.json"))

    update_progress(
        document_id,
        step="understanding_agent_completed",
        message="Understanding agent completed"
    )

    update_progress(
        document_id,
        step="control_logic_pending",
        message="Starting control logic extraction"
    )

    # --- 3. Control Logic Agent ---
    print("\nSTEP 2/5: Running Control Logic Agent...")
    logic_runner = LogicAgentRunner()
    # Pass summary_json (Dict), not a string
    logic_result = logic_runner.run(sections_data, summary_json)
    
    # Extract & Save
    logic_json = extract_json_from_result(logic_result)
    save_json(logic_json, output_dir, "2_logic_extracted.json")
    update_parsed_document(document_id, "control_logic_agent_output_file_path", os.path.join(output_dir, "2_logic_extracted.json"))
    # NOTE: We do NOT convert to string here anymore for the next agent
    # logic_str = json.dumps(logic_json, indent=2) <--- REMOVED

    update_progress(
        document_id,
        step="control_logic_completed",
        message="Control logic extraction completed"
    )

    update_progress(
        document_id,
        step="loop_mapper_pending",
        message="Starting loop mapping"
    )

    # --- 4. Loop Mapper Agent ---
    print("\nSTEP 3/5: Running Loop Mapper Agent...")
    mapper_runner = LoopMapperRunner()
    
    # FIX: Pass the logic_json (Dict) directly
    mapping_result = mapper_runner.run(logic_json)
    
    # Extract & Save
    mapping_json = extract_json_from_result(mapping_result)
    save_json(mapping_json, output_dir, "3_loop_map.json")
    update_parsed_document(document_id, "mapper_agent_output_file_path", os.path.join(output_dir, "3_loop_map.json"))
    update_progress(
        document_id,
        step="loop_mapper_completed",
        message="Loop mapping completed"
    )

    update_progress(
        document_id,
        step="validator_agent_pending",
        message="Starting validator agent"
    )

    # --- 5. Validator Agent ---
    print("\nSTEP 4/5: Running Validator Agent...")
    validator_runner = ValidatorRunner()
    validation_result = validator_runner.run(logic_json, mapping_json)
    
    # Extract & Save
    validation_json = extract_json_from_result(validation_result)
    save_json(validation_json, output_dir, "4_validation.json")
    update_parsed_document(document_id, "validator_agent_output_file_path", os.path.join(output_dir, "4_validation.json"))

    update_progress(
        document_id,
        step="validator_agent_completed",
        message="Validator agent completed"
    )

    # --- 6. Code Generator Agent ---
    print("\nSTEP 5/6: Running Code Generator Agent...")
    codegen_runner = CodeGeneratorRunner()
    
    code_result = codegen_runner.run(logic_json, validation_json)

    final_code = str(code_result)
    save_text(final_code, output_dir, "5_plc_code.st")
    update_parsed_document(document_id, "code_generator_agent_output_file_path", os.path.join(output_dir, "5_plc_code.st"))
    update_progress(
        document_id,
        step="code_generator_completed",
        message="Code generation completed"
    )

    update_progress(
        document_id,
        step="mindmap_generator_agent_pending",
        message="Starting mindmap generation"
    )

    # --- 7. Mindmap Generator Agent ---
    print("\nSTEP 6/6: Running Mindmap Generator Agent...")
    from app.agents.mindmap_agent import MindmapGeneratorRunner # Import here to avoid circular dependencies if any
    mindmap_runner = MindmapGeneratorRunner()
    
    # Run the agent (returns a dict: { "loop_name": {nodes:..., edges:...} })
    mindmap_results = mindmap_runner.run(logic_json)
    
    # Create mindmaps subdirectory
    # Create mindmaps subdirectory
    mindmaps_dir = output_dir / "mindmaps"
    os.makedirs(mindmaps_dir, exist_ok=True)
    
    # Save each loop's mindmap
    saved_files = []
    for loop_name, mm_data in mindmap_results.items():
        # Sanitize filename
        safe_name = "".join([c for c in loop_name if c.isalnum() or c in (' ', '-', '_')]).strip()
        filename = f"{safe_name}.json"
        
        save_json(mm_data, mindmaps_dir, filename)
        saved_files.append({"name": loop_name, "file": filename})

    # Save an index file for easy frontend lookup
    save_json({"mappings": saved_files}, output_dir, "6_mindmaps_index.json")

    update_parsed_document(document_id, "mindmap_generator_output_file_path", os.path.join(output_dir, "6_mindmaps_index.json"))
    
    update_progress(
        document_id,
        step="completed",
        message="Pipeline completed successfully"
    )

    print("\nPIPELINE COMPLETE ✨")
    print(f"All files are saved in: {output_dir}")