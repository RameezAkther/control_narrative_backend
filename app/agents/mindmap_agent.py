
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

# We will use the schema defined in prompt.md, but we need Pydantic models for structured output if possible.
# However, since the output is a complex nested JSON (nodes: list, edges: list) with variable data,
# we might want to ask for a raw JSON string or a simple wrapper.
# Let's try to define a loose structure to help the LLM.

class MindmapNodeData(BaseModel):
    id: str
    type: str
    label: str
    status: str
    currentValue: Union[float, str, int]
    meta: Dict[str, Any] = Field(default_factory=dict)

class MindmapNode(BaseModel):
    id: str
    type: str = "custom"
    position: Dict[str, int] = Field(default_factory=lambda: {"x": 0, "y": 0})
    data: MindmapNodeData

class MindmapEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str = "packetEdge"
    label: str = ""
    animated: bool = True
    style: Dict[str, Any] = Field(default_factory=dict)

class MindmapSystemData(BaseModel):
    nodes: List[MindmapNode]
    edges: List[MindmapEdge]

# ==============================================================================
# 2. The Agent Runner
# ==============================================================================

class MindmapGeneratorRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0.2, # Low temp for consistent JSON structure
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "prompt.md")

    def _load_prompt_instructions(self) -> str:
        """Loads the prompt instructions from the markdown file."""
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompt file not found at: {self.prompt_path}")
        
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def run(self, logic_data: Union[Dict, str]) -> Dict[str, Any]:
        """
        Args:
            logic_data: The dictionary output from LogicAgentRunner (containing "loops").
        Returns:
            A dictionary where keys are loop names and values are the generated Mindmap JSON objects.
        """
        
        # 1. Validate and Parse Input
        if isinstance(logic_data, str):
            try:
                logic_data = json.loads(logic_data)
            except json.JSONDecodeError:
                print("❌ Mindmap Agent Error: Input string is not valid JSON.")
                return {}

        all_loops = logic_data.get("loops", [])
        
        if not all_loops:
            print("⚠️ Mindmap Agent: No loops provided.")
            return {}

        print(f"🔄 Mindmap Agent: Received {len(all_loops)} loops. Generating mindmaps...")

        # Load specific instructions from the file
        try:
            schema_instructions = self._load_prompt_instructions()
        except Exception as e:
            print(f"❌ Mindmap Agent Failed to load prompt: {e}")
            return {}

        # 2. Define Agent
        visualizer = Agent(
            role='Control System Visualizer',
            goal='Convert control loop logic into a ReactFlow-compatible JSON structure.',
            backstory=(
                "You are an expert Frontend Data Architect specializing in Industrial HMI visualization. "
                "You transform abstract control logic (sensors, controllers, pumps) into "
                "structured JSON datasets that drive interactive diagrams."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        results = {}

        # 3. Process Loops (Batching or Iterative)
        # For better quality/context separation, we process each loop individually.
        # This might be slower but ensures one mindmap per loop as requested.
        
        for i, loop in enumerate(all_loops):
            loop_name = loop.get("loop_name", f"Loop_{i}")
            print(f"   ➤ Generating mindmap for: {loop_name}")
            
            loop_context = json.dumps(loop, indent=2)

            task = Task(
                description=(
                    "Generate a ReactFlow Mindmap JSON for the specific Control Loop provided below.\n"
                    "Use the SCHEMA documentation provided to ensure correct format.\n"
                    "================================================================\n"
                    "SCHEMA & INSTRUCTIONS:\n"
                    f"{schema_instructions}\n"
                    "================================================================\n"
                    "INPUT CONTROL LOOP:\n"
                    f"{loop_context}\n"
                    "================================================================\n\n"
                    "**TASK:**\n"
                    "1. Identify every component (Sensor, PID, Pump, Valve, Interlock) in the loop.\n"
                    "2. Create a 'node' for each component using the 'custom' type and appropriate 'data' fields.\n"
                    "3. Create 'edges' (packetEdge) to show the signal flow (e.g. Sensor -> PID -> Valve).\n"
                    "4. Return ONLY the valid JSON object (`nodes` and `edges`)."
                ),
                expected_output="A valid JSON object adhering to the provided schema.",
                agent=visualizer,
                output_json=MindmapSystemData # Enforce Pydantic structure
            )

            crew = Crew(
                agents=[visualizer],
                tasks=[task],
                verbose=False # Reduce noise for individual loops
            )

            try:
                result = crew.kickoff()
                
                # Extract JSON
                mindmap_json = {}
                if hasattr(result, 'json_dict') and result.json_dict:
                    mindmap_json = result.json_dict
                elif hasattr(result, 'pydantic') and result.pydantic:
                    mindmap_json = result.pydantic.model_dump()
                elif hasattr(result, 'raw'):
                     # Fallback cleanup
                    clean = str(result.raw).replace('```json', '').replace('```', '').strip()
                    try:
                        mindmap_json = json.loads(clean)
                    except:
                        print(f"   ❌ Failed to parse JSON for {loop_name}")
                        continue
                
                results[loop_name] = mindmap_json

            except Exception as e:
                print(f"   ❌ Failed loop {loop_name}: {e}")

        return results
