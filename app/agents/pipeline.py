import os

# Import Agents
from app.agents.understanding_agent import UnderstandingAgentRunner, parse_markdown_file
from app.agents.control_logic_agent import LogicAgentRunner
from app.agents.mapper_agent import LoopMapperRunner
from app.agents.validator_agent import ValidatorRunner
from app.agents.code_generator_agent import CodeGeneratorRunner

# Import Utils
from app.utils.io_manager import create_output_dir, save_json, save_text, extract_json_from_result
from app.documents.parsed_crud import update_progress
from app.documents.parsed_crud import update_parsed_document

def run_agent_pipeline(md_file_path: str, document_id: str = None):
    if not os.getenv("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = input("Enter Google API Key: ")

    print(f"\n🚀 STARTING PIPELINE FOR: {md_file_path}")
    
    # Create the structured output folder
    output_dir = create_output_dir(md_file_path)

    # Parse the Markdown Document
    try:
        sections_data = parse_markdown_file(md_file_path)
        print(f"✅ Document Parsed: {len(sections_data)} sections found.")
    except Exception as e:
        print(f"❌ Error parsing file: {e}")
        return
    
    update_progress(
        document_id,
        step="understanding_agent_pending",
        message="Starting understanding agent"
    )

    # --- 2. Understanding Agent ---
    print("\n🧠 STEP 1/5: Running Understanding Agent...")
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
    print("\n🔍 STEP 2/5: Running Control Logic Agent...")
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
    print("\n🗺️ STEP 3/5: Running Loop Mapper Agent...")
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
    print("\n🛡️ STEP 4/5: Running Validator Agent...")
    validator_runner = ValidatorRunner()
    
    # FIX: Pass Dicts directly (logic_json and mapping_json)
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
    print("\n💻 STEP 5/5: Running Code Generator Agent...")
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

    print("\n✨ PIPELINE COMPLETE ✨")
    print(f"All files are saved in: {output_dir}")