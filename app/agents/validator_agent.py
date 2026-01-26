import os
import json
import time
from typing import List, Dict
from pydantic import BaseModel, Field

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. Output Schema (Unchanged)
# ==============================================================================

class ValidationIssue(BaseModel):
    severity: str = Field(..., description="'Error' for blocking issues (missing tags), 'Warning' for minor ones.")
    loop_name: str = Field(..., description="The name of the loop where the issue was found.")
    message: str = Field(..., description="Description of the issue.")
    suggestion: str = Field(..., description="Recommended fix.")

class ValidationReport(BaseModel):
    # We remove 'is_valid' here because we will calculate it based on the issue list size later
    issues: List[ValidationIssue] = Field(default_factory=list, description="List of all identified issues.")

# ==============================================================================
# 2. The Agent Runner (Robust Batching Version)
# ==============================================================================

class ValidatorRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _merge_data(self, logic_loops: List[Dict], mapped_loops: List[Dict]) -> List[Dict]:
        merged = []
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

    def _batch_data(self, data: List[Dict], batch_size=5) -> List[List[Dict]]:
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def run(self, logic_data: Dict, mapping_data: Dict) -> Dict:
        l_loops = logic_data.get('loops', [])
        m_mappings = mapping_data.get('mappings', [])

        if not l_loops:
            return {"is_valid": True, "issues": [], "summary": "No loops to validate."}

        full_loop_contexts = self._merge_data(l_loops, m_mappings)
        
        batches = self._batch_data(full_loop_contexts, batch_size=5)
        print(f"🕵️ Validator: Checking {len(full_loop_contexts)} loops across {len(batches)} batches.")

        agent = Agent(
            role='Control Systems QA Engineer',
            goal='Validate control logic against best practices.',
            backstory="You are the final gatekeeper. You check for missing interlocks, undefined tags, and logic gaps.",
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )

        all_issues = []

        # --- PROCESS BATCHES ---
        for i, batch in enumerate(batches):
            print(f"   Validating Batch {i+1}/{len(batches)}...")
            
            batch_str = json.dumps(batch, indent=2)
            
            task = Task(
                description=(
                    f"Validate this BATCH of control loops:\n{batch_str}\n\n"
                    "CHECK FOR:\n"
                    "1. Missing Sensors/Actuators for the described strategy.\n"
                    "2. Critical loops missing safety interlocks.\n"
                    "3. Vagueness in descriptions.\n"
                    "Return a JSON list of issues found."
                ),
                expected_output="JSON list of validation issues.",
                agent=agent,
                output_json=ValidationReport
            )

            # Isolate task
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )

            try:
                result = crew.kickoff()
                
                # Robust extraction
                batch_report = None
                if hasattr(result, 'json_dict') and result.json_dict:
                    batch_report = result.json_dict
                elif hasattr(result, 'pydantic') and result.pydantic:
                    batch_report = result.pydantic.dict()
                elif hasattr(result, 'raw'):
                    try:
                        clean = result.raw.replace('```json', '').replace('```', '')
                        batch_report = json.loads(clean)
                    except:
                        pass

                if batch_report and 'issues' in batch_report:
                    count = len(batch_report['issues'])
                    if count > 0:
                        print(f"   ⚠️ Batch {i+1}: Found {count} issues.")
                        all_issues.extend(batch_report['issues'])
                    else:
                        print(f"   ✅ Batch {i+1}: Clean.")
                else:
                    print(f"   ⚠️ Batch {i+1}: No valid output structure.")

            except Exception as e:
                print(f"   ❌ Error validating Batch {i+1}: {e}")

            # RATE LIMITING
            time.sleep(2)

        # --- FINAL AGGREGATION ---
        final_report = {
            "issues": all_issues,
            "is_valid": len(all_issues) == 0,
            "total_checked": len(full_loop_contexts)
        }
        
        return final_report