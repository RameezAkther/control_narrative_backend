import os
import json
import time
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field

# CrewAI imports
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. Output Schema
# ==============================================================================

class ValidationIssue(BaseModel):
    severity: str = Field(..., description="Level: 'Critical' (Safety risk), 'Error' (Missing component), 'Warning' (Best practice).")
    loop_name: str = Field(..., description="The name of the loop where the issue was found.")
    message: str = Field(..., description="Clear description of the problem.")
    suggestion: str = Field(..., description="Actionable fix (e.g. 'Add Low Level Interlock to P-101').")

class ValidationReport(BaseModel):
    issues: List[ValidationIssue] = Field(default_factory=list, description="List of all identified issues.")

# ==============================================================================
# 2. The Validator Runner (Single-Shot Efficiency)
# ==============================================================================

class ValidatorRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0, # Zero temp for strict rule checking
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Safety limit for Single-Shot
        self.MAX_LOOPS_SINGLE_SHOT = 2000

    def _merge_data(self, logic_loops: List[Dict], mapped_loops: List[Dict]) -> List[Dict]:
        """
        Combines Logic (Sensors/Actuators) with Mapping (Strategy/Criticality)
        to create a rich context for validation.
        """
        merged = []
        # Create a lookup map for efficiency
        mapping_map = {m.get('loop_name'): m for m in mapped_loops}
        
        for logic in logic_loops:
            l_name = logic.get('loop_name')
            mapping = mapping_map.get(l_name, {})
            
            merged_item = {
                "loop_id": l_name,
                "components": {
                    "sensors": logic.get('inputs', []),
                    "actuators": logic.get('outputs', []),
                    "interlocks": logic.get('interlocks', [])
                },
                "strategy_info": {
                    "type": mapping.get('strategy_type', 'Unknown'),
                    "criticality": mapping.get('criticality', 'Unknown'),
                    "topology": mapping.get('topology_description', '')
                }
            }
            merged.append(merged_item)
        return merged

    def run(self, logic_data: Union[Dict, str], mapping_data: Union[Dict, str]) -> Dict[str, Any]:
        """
        Validates the entire control system in one pass.
        """
        
        # 1. Input Parsing
        if isinstance(logic_data, str): logic_data = json.loads(logic_data)
        if isinstance(mapping_data, str): mapping_data = json.loads(mapping_data)

        l_loops = logic_data.get('loops', [])
        m_mappings = mapping_data.get('mappings', [])

        if not l_loops:
            print("⚠️ Validator: No loops to validate.")
            return {"is_valid": True, "issues": [], "summary": "No data provided."}

        # 2. Merge Data for Full Context
        full_system_context = self._merge_data(l_loops, m_mappings)
        loop_count = len(full_system_context)
        
        print(f"🕵️ Validator: Analyzing {loop_count} loops in SINGLE-SHOT mode...")

        # 3. Check Size Strategy
        if loop_count > self.MAX_LOOPS_SINGLE_SHOT:
            print(f"⚠️ Truncating {loop_count} loops to {self.MAX_LOOPS_SINGLE_SHOT} for safety.")
            full_system_context = full_system_context[:self.MAX_LOOPS_SINGLE_SHOT]

        # 4. Define Agent
        qa_engineer = Agent(
            role='Control Systems QA Lead',
            goal='Audit the Control Narrative for completeness, safety, and logic gaps.',
            backstory=(
                "You are a strict QA Auditor. You check if High Criticality loops have Safety Interlocks. "
                "You check if PID loops have both Sensors (Inputs) and Actuators (Outputs). "
                "You flag vague names like 'Unknown Component'."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # 5. Prepare Context
        context_str = json.dumps(full_system_context, indent=2)

        task = Task(
            description=(
                "Audit the following Control System definition:\n"
                "================================================================\n"
                f"{context_str}\n"
                "================================================================\n\n"
                "**VALIDATION RULES:**\n"
                "1. **Completeness**: A PID Loop MUST have at least one Sensor and one Actuator.\n"
                "2. **Safety**: Any loop marked 'Criticality: High' MUST have an Interlock or Safety Logic defined.\n"
                "3. **Clarity**: Flag any component named 'Unknown', 'Unnamed', or 'TBD'.\n"
                "4. **Consistency**: Ensure the Strategy Type matches the components (e.g., Don't call it 'PID' if it only has a switch).\n\n"
                "**OUTPUT:**\n"
                "Return a list of `ValidationIssue` objects for any violations found."
            ),
            expected_output="A ValidationReport JSON object.",
            agent=qa_engineer,
            output_json=ValidationReport
        )

        # 6. Execute
        crew = Crew(
            agents=[qa_engineer],
            tasks=[task],
            verbose=True
        )

        try:
            print("🚀 Sending system audit request...")
            start_time = time.time()
            result = crew.kickoff()
            elapsed = time.time() - start_time
            print(f"✅ Validation complete in {elapsed:.2f} seconds.")

            # Robust Extraction
            report_data = {"issues": []}
            if hasattr(result, 'json_dict') and result.json_dict:
                report_data = result.json_dict
            elif hasattr(result, 'pydantic') and result.pydantic:
                report_data = result.pydantic.model_dump()
            elif hasattr(result, 'raw'):
                try:
                    clean = result.raw.replace('```json', '').replace('```', '').strip()
                    report_data = json.loads(clean)
                except:
                    pass

            issues = report_data.get("issues", [])
            print(f"🔍 Found {len(issues)} issues.")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "total_checked": loop_count
            }

        except Exception as e:
            print(f"❌ Validation Failed: {e}")
            return {"is_valid": False, "issues": [{"severity": "Error", "loop_name": "System", "message": str(e), "suggestion": "Check logs."}]}

# ==============================================================================
# Usage Example
# ==============================================================================
# if __name__ == "__main__":
#     # Mock Data
#     logic_mock = {"loops": [{"loop_name": "P-101", "inputs": [], "outputs": [{"tag": "P-101"}]}]}
#     mapping_mock = {"mappings": [{"loop_name": "P-101", "strategy_type": "PID Control", "criticality": "High"}]}
#     
#     runner = ValidatorRunner()
#     report = runner.run(logic_mock, mapping_mock)
#     print(json.dumps(report, indent=2))