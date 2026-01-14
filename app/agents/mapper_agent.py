import os
import json
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
# 2. The Agent Runner (Batched Version)
# ==============================================================================

class LoopMapperRunner:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _batch_loops(self, loops: List[Dict], batch_size=10) -> List[List[Dict]]:
        """Helper to split large list of loops into smaller batches."""
        return [loops[i:i + batch_size] for i in range(0, len(loops), batch_size)]

    def run(self, logic_data: Dict[str, Any]) -> Dict:
        """
        Args:
            logic_data: The raw dictionary output from LogicAgentRunner (NOT a string).
                        Expected format: {'loops': [...]}
        """
        
        # 1. Validate Input
        # If logic_data is a string (rare), parse it. Otherwise assume dict.
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

        print(f"🔄 Processing {len(all_loops)} loops in batches...")

        # 2. Batch the loops
        loop_batches = self._batch_loops(all_loops, batch_size=5) # Small batch = high accuracy

        # 3. Define Agent
        agent = Agent(
            role='Control Topology Architect',
            goal='Determine control strategies and visualize I/O relationships for a specific set of loops.',
            backstory="You are a systems architect. You look at raw sensor/actuator lists and determine if they represent a PID loop, a Safety Interlock, or a Sequence.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # 4. Create Tasks (One per batch)
        tasks = []
        for i, batch in enumerate(loop_batches):
            # We convert the batch to a pretty JSON string for the prompt
            batch_str = json.dumps(batch, indent=2)
            
            task = Task(
                description=(
                    f"Analyze this BATCH of Control Loops (Batch {i+1}/{len(loop_batches)}):\n\n"
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
            tasks.append(task)

        # 5. Run Crew
        # Note: We do NOT need a 'Reduce' agent here because we just want to concatenate the lists.
        # We can do that in Python after the crew finishes.
        crew = Crew(
            agents=[agent],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        crew_output = crew.kickoff()

        # 6. Aggregation (Manual Reduce)
        # Since CrewAI returns the *final* task output by default, we need to inspect the individual task outputs 
        # or rely on the fact that we are processing sequentially. 
        # However, a safer way in CrewAI for lists is to collect results.
        
        # *Self-Correction*: The simplest way to handle aggregation in this specific pattern 
        # without a complex Reduce agent is to iterate the results manually or use a Reduce agent.
        # Let's add a simple Reduce agent to make it cleaner and strictly return one JSON.
        
        reducer_agent = Agent(
            role='Mapping Aggregator',
            goal='Combine batch results into one final list.',
            backstory="You simply take multiple lists of mappings and combine them into one final JSON object.",
            llm=self.llm,
            verbose=True
        )

        reduce_task = Task(
            description=("Combine all the partial mapping lists from previous tasks into one single 'mappings' list."),
            expected_output="The final merged MappingResult JSON.",
            agent=reducer_agent,
            context=tasks, # <--- This gives the reducer access to all batch outputs
            output_json=MappingResult
        )

        final_crew = Crew(
            agents=[agent, reducer_agent],
            tasks=[*tasks, reduce_task],
            process=Process.sequential,
            verbose=True
        )

        result = final_crew.kickoff()

        if hasattr(result, 'json_dict') and result.json_dict:
            return result.json_dict
        elif hasattr(result, 'raw'):
             try:
                clean_raw = result.raw.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_raw)
             except:
                return {"mappings": []}
        
        return result