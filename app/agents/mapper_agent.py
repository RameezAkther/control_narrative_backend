import os
import json
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. Output Schema
# ==============================================================================

class MappedRelationship(BaseModel):
    loop_name: str = Field(..., description="Name of the loop being mapped.")
    strategy_type: str = Field(..., description="Control Strategy: 'PID Control', 'On/Off Sequence', 'Cascade', 'Ratio Control', or 'Safety Interlock'.")
    topology_description: str = Field(..., description="A clear text description of the signal flow (e.g. 'Sensor X triggers Valve Y').")
    criticality: str = Field(..., description="High, Medium, or Low.")

class MappingResult(BaseModel):
    mappings: List[MappedRelationship] = Field(..., description="List of mapped relationships.")

# ==============================================================================
# 2. The Agent Runner (Robust Batching Version)
# ==============================================================================

class LoopMapperRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _batch_loops(self, loops: List[Dict], batch_size=5) -> List[List[Dict]]:
        """Helper to split large list of loops into smaller batches."""
        return [loops[i:i + batch_size] for i in range(0, len(loops), batch_size)]

    def run(self, logic_data: Dict[str, Any]) -> Dict:
        """
        Args:
            logic_data: The raw dictionary output from LogicAgentRunner.
        """
        
        # 1. Validate Input
        if isinstance(logic_data, str):
            try:
                logic_data = json.loads(logic_data)
            except json.JSONDecodeError:
                print("❌ Mapper Error: Input was a string but invalid JSON.")
                return {"mappings": []}

        all_loops = logic_data.get("loops", [])
        if not all_loops:
            print("⚠️ No loops found to map.")
            return {"mappings": []}

        # 2. Batch the loops
        # Batch size of 5 is safe for output token limits and context
        loop_batches = self._batch_loops(all_loops, batch_size=5) 
        print(f"🔄 Mapper: Processing {len(all_loops)} loops across {len(loop_batches)} batches...")

        # 3. Define Agent
        agent = Agent(
            role='Control Topology Architect',
            goal='Determine control strategies and visualize I/O relationships.',
            backstory="You are a systems architect. You determine if a loop is PID, Safety Interlock, or Sequence.",
            llm=self.llm,
            verbose=False, # Reduce noise
            allow_delegation=False
        )

        # 4. Process Batches Sequentially (Map Phase)
        all_mappings = []

        for i, batch in enumerate(loop_batches):
            print(f"   Processing Batch {i+1}/{len(loop_batches)}...")
            
            # Convert batch to string for the prompt
            batch_str = json.dumps(batch, indent=2)
            
            task = Task(
                description=(
                    f"Analyze this BATCH of Control Loops:\n\n"
                    f"{batch_str}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. For EACH loop in the list above, determine the 'strategy_type'.\n"
                    "2. Write a 'topology_description' explaining the flow.\n"
                    "3. Assign 'criticality' based on the application.\n"
                    "4. Return the result as a structured JSON list."
                ),
                expected_output="A JSON list of MappedRelationships.",
                agent=agent,
                output_json=MappingResult 
            )

            # Isolate task in a mini-crew
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )

            try:
                result = crew.kickoff()
                
                # Robust extraction
                batch_result = None
                if hasattr(result, 'json_dict') and result.json_dict:
                    batch_result = result.json_dict
                elif hasattr(result, 'pydantic') and result.pydantic:
                    batch_result = result.pydantic.dict()
                elif hasattr(result, 'raw'):
                    try:
                        clean = result.raw.replace('```json', '').replace('```', '')
                        batch_result = json.loads(clean)
                    except:
                        pass

                if batch_result and 'mappings' in batch_result:
                    count = len(batch_result['mappings'])
                    print(f"   ✅ Batch {i+1}: Mapped {count} loops.")
                    all_mappings.extend(batch_result['mappings'])
                else:
                    print(f"   ⚠️ Batch {i+1}: No mappings found or invalid output.")

            except Exception as e:
                print(f"   ❌ Error processing Batch {i+1}: {e}")

            # RATE LIMITING: Sleep to respect API limits
            time.sleep(2)

        # 5. Final Aggregation
        print(f"✨ Mapper Complete. Total mappings: {len(all_mappings)}")
        
        return {"mappings": all_mappings}