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

class MappedRelationship(BaseModel):
    loop_name: str = Field(..., description="Name of the loop being mapped.")
    strategy_type: str = Field(..., description="The Control Strategy: 'PID Control', 'On/Off Sequence', 'Cascade', 'Ratio Control', 'Safety Interlock', or 'Monitoring Only'.")
    topology_description: str = Field(..., description="A concise technical description of the signal flow (e.g. 'LIT-101 (PV) -> LIC-101 (PID) -> LCV-101 (MV)').")
    criticality: str = Field(..., description="Assessment: 'High' (Safety/Shutdown), 'Medium' (Process Control), or 'Low' (Monitoring).")

class MappingResult(BaseModel):
    mappings: List[MappedRelationship] = Field(..., description="List of mapped relationships.")

# ==============================================================================
# 2. The Agent Runner (Single-Shot High Efficiency)
# ==============================================================================

class LoopMapperRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0.1, # Low temp for consistent classification
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Safety limit: 1M tokens can easily hold 5,000+ loops.
        # We set a logical limit just to be safe.
        self.MAX_LOOPS_SINGLE_SHOT = 2000

    def run(self, logic_data: Union[Dict, str]) -> Dict[str, Any]:
        """
        Args:
            logic_data: The dictionary output from LogicAgentRunner (containing "loops").
        """
        
        # 1. Validate and Parse Input
        if isinstance(logic_data, str):
            try:
                logic_data = json.loads(logic_data)
            except json.JSONDecodeError:
                print("❌ Mapper Error: Input string is not valid JSON.")
                return {"mappings": []}

        all_loops = logic_data.get("loops", [])
        
        if not all_loops:
            print("⚠️ Mapper: No loops provided to map.")
            return {"mappings": []}

        print(f"🔄 Mapper: Received {len(all_loops)} loops. Processing in SINGLE-SHOT mode...")

        # 2. Check Size Strategy
        if len(all_loops) > self.MAX_LOOPS_SINGLE_SHOT:
            print(f"⚠️ Too many loops ({len(all_loops)}). Truncating to {self.MAX_LOOPS_SINGLE_SHOT} for safety.")
            all_loops = all_loops[:self.MAX_LOOPS_SINGLE_SHOT]

        # 3. Define Agent
        architect = Agent(
            role='Control Topology Architect',
            goal='Analyze a list of Control Loops and determine their Control Strategy and Criticality.',
            backstory=(
                "You are a Senior Systems Architect. You look at raw inputs/outputs and determine "
                "the underlying strategy. "
                "Example: If you see a Level Transmitter and a Control Valve, it's likely 'PID Control'. "
                "Example: If you see a 'High High Level' triggering a 'Pump Trip', it is a 'Safety Interlock'."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # 4. Prepare Context
        # We dump the entire list of loops into the prompt.
        loops_context = json.dumps(all_loops, indent=2)

        task = Task(
            description=(
                "Analyze the following list of Control Loops and map their strategies.\n"
                "================================================================\n"
                f"{loops_context}\n"
                "================================================================\n\n"
                "**INSTRUCTIONS:**\n"
                "1. **Classify Strategy**: Determine if it is PID, Sequence, Interlock, etc.\n"
                "2. **Describe Topology**: Briefly explain the signal flow (Sensor -> Controller -> Actuator).\n"
                "3. **Assess Criticality**: Mark Safety/Shutdown loops as 'High'.\n"
                "4. **Output**: Return a JSON object with the list of mappings."
            ),
            expected_output="A complete MappingResult JSON object.",
            agent=architect,
            output_json=MappingResult
        )

        # 5. Execute
        crew = Crew(
            agents=[architect],
            tasks=[task],
            verbose=True
        )

        try:
            print("🚀 Sending all loops to Gemini...")
            start_time = time.time()
            result = crew.kickoff()
            elapsed = time.time() - start_time
            print(f"✅ Mapping complete in {elapsed:.2f} seconds.")

            # Robust Extraction
            if hasattr(result, 'json_dict') and result.json_dict:
                return result.json_dict
            elif hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic.model_dump()
            elif hasattr(result, 'raw'):
                try:
                    clean = result.raw.replace('```json', '').replace('```', '').strip()
                    return json.loads(clean)
                except:
                    print("❌ JSON Parse failed on raw output.")
                    
            return {"mappings": []}

        except Exception as e:
            print(f"❌ Mapper Failed: {e}")
            return {"mappings": []}

# ==============================================================================
# Usage Example
# ==============================================================================
# if __name__ == "__main__":
#     # Mock input from Logic Agent
#     mock_logic = {
#         "loops": [
#             {"loop_name": "Feed Pump Control", "inputs": [{"tag": "LIT-101"}], "outputs": [{"tag": "P-101"}]},
#             {"loop_name": "Emergency Stop", "inputs": [{"tag": "ESD-001"}], "outputs": [{"tag": "All Systems"}]}
#         ]
#     }
#     runner = LoopMapperRunner()
#     res = runner.run(mock_logic)
#     print(json.dumps(res, indent=2))