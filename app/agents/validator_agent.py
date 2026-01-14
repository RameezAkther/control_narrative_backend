import os
import json
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
# 2. The Agent Runner (Batched & Merged)
# ==============================================================================

class ValidatorRunner:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
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
            goal='Validate control logic.',
            backstory="You are the final gatekeeper.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        tasks = []
        for i, batch in enumerate(batches):
            batch_str = json.dumps(batch, indent=2)
            task = Task(
                description=f"Validate Batch {i+1}:\n{batch_str}",
                expected_output="JSON list of validation issues.",
                agent=agent,
                output_json=ValidationReport
            )
            tasks.append(task)

        aggregator = Agent(
            role='QA Lead',
            goal='Compile final report.',
            backstory="Compile punch-list.",
            llm=self.llm,
            verbose=True
        )

        reduce_task = Task(
            description="Combine issues into one list.",
            expected_output="Final ValidationReport.",
            agent=aggregator,
            context=tasks,
            output_json=ValidationReport
        )

        crew = Crew(
            agents=[agent, aggregator],
            tasks=[*tasks, reduce_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        
        final_report = {}
        
        if hasattr(result, 'json_dict') and result.json_dict:
            final_report = result.json_dict
        elif hasattr(result, 'raw'):
            try:
                clean_raw = result.raw.replace('```json', '').replace('```', '').strip()
                final_report = json.loads(clean_raw)
            except:
                print(f"❌ JSON Parse Error on raw output: {result.raw}")
                final_report = {"issues": []}
        else:
            final_report = result if isinstance(result, dict) else {"issues": []}

        is_valid = len(final_report.get('issues', [])) == 0
        final_report['is_valid'] = is_valid
        
        return final_report