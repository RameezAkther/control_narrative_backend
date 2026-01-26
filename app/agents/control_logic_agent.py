import os
import json
import re
import time
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. Output Schema (Unchanged)
# ==============================================================================

class ControlElement(BaseModel):
    tag: Optional[str] = Field(default="Unknown", description="The instrument tag (e.g., 'LIT-101').")
    name: Optional[str] = Field(default="Unnamed Component", description="The name of the element.")
    role: Optional[str] = Field(default="Unknown", description="Role: 'Sensor', 'Actuator', etc.")

class Interlock(BaseModel):
    condition: Optional[str] = Field(default="Unspecified condition", description="The logical condition.")
    action: Optional[str] = Field(default="Unspecified action", description="The action taken.")
    type: Optional[str] = Field(default="Process", description="Type: 'Safety', 'Process', etc.")

class ControlLoop(BaseModel):
    loop_name: Optional[str] = Field(default="Unnamed Loop", description="Name of the loop.")
    description: Optional[str] = Field(default="No description provided.", description="Brief description.")
    inputs: List[ControlElement] = Field(default_factory=list, description="List of sensors/inputs.")
    outputs: List[ControlElement] = Field(default_factory=list, description="List of actuators/outputs.")
    interlocks: List[Interlock] = Field(default_factory=list, description="List of specific logic conditions.")

class LogicExtractionResult(BaseModel):
    loops: List[ControlLoop] = Field(default_factory=list, description="A list of all identified control loops.")

# ==============================================================================
# 2. The Agent Runner (Fault-Tolerant Batching)
# ==============================================================================

class LogicAgentRunner:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _create_chunks(self, sections: List[Dict], max_chunk_size=15000) -> List[str]:
        chunks = []
        current_chunk = ""
        for sec in sections:
            sec_text = f"\n--- SECTION: {sec.get('title', 'Unknown')} ---\n"
            sec_text += sec.get('content', '') + "\n"
            if len(current_chunk) + len(sec_text) > max_chunk_size:
                if current_chunk: chunks.append(current_chunk)
                current_chunk = sec_text
            else:
                current_chunk += sec_text
        if current_chunk: chunks.append(current_chunk)
        return chunks

    def _clean_json_string(self, raw_text: str) -> str:
        """Helper to extract valid JSON from LLM chatter."""
        try:
            # 1. Try finding content inside ```json ... ```
            match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if match: return match.group(1)
            
            # 2. Try finding content inside ``` ... ```
            match = re.search(r"```\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
            if match: return match.group(1)

            # 3. Try finding first { and last }
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start != -1 and end != -1:
                return raw_text[start:end+1]
                
            return raw_text
        except:
            return raw_text

    def run(self, sections: List[Dict], summary_context: Dict = None):
        doc_chunks = self._create_chunks(sections)
        print(f"📄 Logic Extraction: Document split into {len(doc_chunks)} chunks.")

        summary_str = ""
        if summary_context:
            summary_str = f"SYSTEM OVERVIEW CONTEXT:\n{json.dumps(summary_context, indent=2)}\n"

        logic_analyst = Agent(
            role='Control Logic Analyst',
            goal='Extract exact Control Loops, I/O tags, and Logic from a text fragment.',
            backstory="You are a detail-oriented engineer. You identify every control loop mentioned on the page.",
            llm=self.llm,
            allow_delegation=False,
            verbose=False
        )

        lead_automation_eng = Agent(
            role='Lead Automation Engineer',
            goal='Merge multiple lists of control loops into one robust Master List.',
            backstory="You receive partial lists from your team and de-duplicate them.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

        # --- PHASE 1: MAP (Resilient Processing) ---
        all_partial_results = []
        print("\n🔄 Starting Batch Extraction (Map Phase)...")
        
        for i, chunk_text in enumerate(doc_chunks):
            print(f"   Processing Chunk {i+1}/{len(doc_chunks)}...")
            
            task = Task(
                description=(
                    f"Analyze this PARTIAL text (Chunk {i+1}/{len(doc_chunks)}) to find Control Logic.\n\n"
                    f"{summary_str}\n"
                    "TEXT TO ANALYZE:\n"
                    f"{chunk_text}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Identify Control Loops visible in THIS text only.\n"
                    "2. Extract Sensors (Inputs) and Actuators (Outputs).\n"
                    "3. Extract Interlock/Logic conditions.\n"
                    "4. Return a JSON object with a 'loops' key containing the list."
                ),
                expected_output="A valid JSON object containing extracted control loops.",
                agent=logic_analyst
                # REMOVED output_json to prevent auto-crash
            )

            crew = Crew(agents=[logic_analyst], tasks=[task], verbose=False)

            try:
                result = crew.kickoff()
                
                # Manual Extraction logic to handle bad JSON gracefully
                raw_output = str(result)
                if hasattr(result, 'raw'): raw_output = result.raw
                
                clean_str = self._clean_json_string(raw_output)
                
                # Parse
                data = json.loads(clean_str)
                
                # Validate using Pydantic manually
                validated = LogicExtractionResult(**data)
                
                if validated.loops:
                    print(f"   ✅ Chunk {i+1}: Found {len(validated.loops)} loops.")
                    all_partial_results.extend(validated.loops)
                else:
                    print(f"   ⚠️ Chunk {i+1}: No loops found (Empty).")

            except json.JSONDecodeError:
                print(f"   ❌ Chunk {i+1} Failed: Invalid JSON output from LLM. Skipping chunk.")
            except Exception as e:
                print(f"   ❌ Chunk {i+1} Failed: {str(e)[:100]}... Skipping chunk.")

            time.sleep(2) 

        # --- PHASE 2: REDUCE ---
        print(f"\n🧩 Starting Aggregation (Reduce Phase) with {len(all_partial_results)} raw loops...")

        if not all_partial_results:
            return {"loops": []}

        # Convert Pydantic models to dicts for JSON serialization
        loops_data = [l.model_dump() for l in all_partial_results]
        
        # Batch the reducer if too many loops (avoid context limit)
        # For simplicity, we assume < 500 loops fits in Gemini's massive window.
        loops_context = json.dumps(loops_data, indent=2)

        reduce_task = Task(
            description=(
                "Merge and Deduplicate these Control Loops.\n\n"
                f"RAW DATA:\n{loops_context}\n\n"
                "RULES:\n"
                "1. Merge duplicate loops (e.g. 'Heater 1' vs 'Heater 1 Control').\n"
                "2. Remove duplicate tags in inputs/outputs.\n"
                "3. Return the final list as JSON."
            ),
            expected_output="The final merged LogicExtractionResult JSON.",
            agent=lead_automation_eng,
            output_json=LogicExtractionResult # We can keep strict mode here as input is cleaner
        )

        final_crew = Crew(
            agents=[lead_automation_eng],
            tasks=[reduce_task],
            verbose=True
        )

        try:
            final_result = final_crew.kickoff()
            if hasattr(final_result, 'json_dict') and final_result.json_dict:
                return final_result.json_dict
            elif hasattr(final_result, 'pydantic') and final_result.pydantic:
                return final_result.pydantic.model_dump()
        except Exception as e:
            print(f"❌ Reducer Failed: {e}. Returning raw aggregated list.")
            return {"loops": loops_data}
            
        return {"loops": []}