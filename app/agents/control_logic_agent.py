import os
import json
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
# 2. The Agent Runner (Map-Reduce Version)
# ==============================================================================

class LogicAgentRunner:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _create_chunks(self, sections: List[Dict], max_chunk_size=30000) -> List[str]:
        """
        Groups sections into larger chunks to avoid creating too many small tasks.
        """
        chunks = []
        current_chunk = ""
        
        for sec in sections:
            sec_text = f"\n--- SECTION: {sec.get('title', 'Unknown')} ---\n"
            sec_text += sec.get('content', '') + "\n"
            
            if len(current_chunk) + len(sec_text) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sec_text
            else:
                current_chunk += sec_text
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def run(self, sections: List[Dict], summary_context: Dict = None):
        
        # 1. Chunk the document
        doc_chunks = self._create_chunks(sections)
        print(f"📄 Logic Extraction: Document split into {len(doc_chunks)} chunks.")

        # 2. Prepare Summary Context (Stringified once to pass to all agents)
        summary_str = ""
        if summary_context:
            summary_str = f"SYSTEM OVERVIEW CONTEXT:\n{json.dumps(summary_context, indent=2)}\n"

        # --- AGENTS ---

        # The Mapper: Focuses on finding loops in a small text window
        logic_analyst = Agent(
            role='Control Logic Analyst',
            goal='Extract exact Control Loops, I/O tags, and Logic from a text fragment.',
            backstory=(
                "You are a detail-oriented engineer reading a single page of a specification. "
                "You identify every control loop mentioned on that page. "
                "You do not guess about what is happening on other pages."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

        # The Reducer: Merges lists and handles duplicates
        lead_automation_eng = Agent(
            role='Lead Automation Engineer',
            goal='Merge multiple lists of control loops into one robust Master List.',
            backstory=(
                "You receive partial lists of control loops from your team. "
                "Your job is to de-duplicate them. "
                "If 'Tank Level' is mentioned in Chunk 1 and Chunk 2, merge them into a single loop entry. "
                "Ensure tag names are consistent."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

        # --- TASKS ---

        map_tasks = []

        # MAP PHASE: Create a task for each chunk
        for i, chunk_text in enumerate(doc_chunks):
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
                    "4. If a loop seems incomplete, extract what is visible (e.g., just the sensors).\n"
                ),
                expected_output="A partial list of Control Loops found in this chunk.",
                agent=logic_analyst,
                output_json=LogicExtractionResult # Returns a partial list of loops
            )
            map_tasks.append(task)

        # REDUCE PHASE: Merge everything
        reduce_task = Task(
            description=(
                "You have received multiple partial lists of Control Loops. "
                "Merge them into one final consolidated list.\n\n"
                "RULES FOR MERGING:\n"
                "1. **Deduplicate Loops**: If 'Heater Control' appears in multiple chunks, create ONE loop entry and combine their sensors/interlocks.\n"
                "2. **Deduplicate Tags**: Ensure the same tag (e.g., TIT-101) is not listed twice in the same loop.\n"
                "3. **Consolidate Logic**: Combine interlock conditions found in different sections for the same equipment.\n"
                "4. **Final Check**: Ensure the output matches the required JSON structure exactly."
            ),
            expected_output="The final merged LogicExtractionResult containing all Control Loops.",
            agent=lead_automation_eng,
            context=map_tasks, # Passes all partial outputs to this task
            output_json=LogicExtractionResult
        )

        # --- CREW EXECUTION ---

        crew = Crew(
            agents=[logic_analyst, lead_automation_eng],
            tasks=[*map_tasks, reduce_task],
            process=Process.sequential,
            verbose=True
        )

        return crew.kickoff()