import os
import json
import time
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# CrewAI imports
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. Output Schema (Refined for Better Extraction)
# ==============================================================================

class ControlElement(BaseModel):
    tag: str = Field(..., description="The instrument tag (e.g., 'LIT-101', 'P-101'). Do not use generic names.")
    name: str = Field(..., description="The descriptive name found in the document (e.g., 'Feed Tank Level', 'Booster Pump'). Infer from context if not explicit.")
    role: str = Field(..., description="Must be one of: 'Sensor', 'Actuator', 'Setpoint', 'Switch', 'Controller'.")

class Interlock(BaseModel):
    condition: str = Field(..., description="The specific condition causing the action (e.g., 'Level < 20% for 5s').")
    action: str = Field(..., description="The consequence (e.g., 'Trip Pump P-101', 'Inhibit Start').")
    type: str = Field(default="Process", description="Category: 'Safety Interlock', 'Process Interlock', or 'Permissive'.")

class ControlLoop(BaseModel):
    loop_name: str = Field(..., description="A descriptive name for the loop (e.g. 'Feed Tank Level Control').")
    description: str = Field(..., description="Summary of the control strategy (PID, On/Off, Sequence).")
    inputs: List[ControlElement] = Field(default_factory=list, description="Sensors and signals affecting this loop.")
    outputs: List[ControlElement] = Field(default_factory=list, description="Actuators and devices controlled by this loop.")
    interlocks: List[Interlock] = Field(default_factory=list, description="Safety trips, start permissives, and auto-shutdowns.")

class LogicExtractionResult(BaseModel):
    loops: List[ControlLoop] = Field(default_factory=list, description="A comprehensive list of all identified control loops.")

# ==============================================================================
# 2. The Logic Runner (Single-Shot / Hybrid)
# ==============================================================================

class LogicAgentRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0.1, # Low temp for precision
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            safety_settings={
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE", 
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            }
        )
        # Limit for Single-Shot (approx 800k tokens to be safe)
        self.MAX_SINGLE_SHOT_CHARS = 3_200_000 
        self.CHUNK_SIZE_CHARS = 100_000

    def _construct_full_context(self, sections: List[Dict]) -> str:
        """Merges all sections into one massive text block with navigation headers."""
        full_text = []
        for sec in sections:
            title = sec.get('title', 'Untitled')
            content = sec.get('content', '')
            full_text.append(f"\n# SECTION START: {title}")
            full_text.append(content)
            full_text.append(f"# SECTION END: {title}")
            full_text.append("-" * 40)
        return "\n".join(full_text)

    def run(self, sections: List[Dict], summary_context: Dict = None) -> Dict[str, Any]:
        
        # 1. Flatten document
        full_text = self._construct_full_context(sections)
        doc_length = len(full_text)
        
        print(f"📄 Logic Extraction: Document Loaded. Length: {doc_length:,} chars.")

        # 2. Add Summary Context if available (helps ground the model)
        context_str = ""
        if summary_context:
            context_str = f"SYSTEM OVERVIEW (Use for context):\n{json.dumps(summary_context, indent=2)}\n\n"

        # 3. Strategy Selection
        if doc_length < self.MAX_SINGLE_SHOT_CHARS:
            print("🟢 Strategy: SINGLE-SHOT (Best Quality)")
            return self._run_single_shot(full_text, context_str)
        else:
            print("🔴 Strategy: MAP-REDUCE (Too large, falling back)")
            return self._run_map_reduce(full_text, context_str)

    # --------------------------------------------------------------------------
    # Strategy A: Single Shot (The Fix for "Unknown Components")
    # --------------------------------------------------------------------------
    def _run_single_shot(self, text: str, context_prefix: str):
        agent = Agent(
            role='Senior Automation Logic Engineer',
            goal='Extract detailed Control Loops, matching Tags to their Descriptions across the entire document.',
            backstory=(
                "You are an expert FDS analyst. "
                "CRITICAL: When you find a tag like 'PT-1000' in a logic sentence, "
                "you IMMEDIATELY look back at the Instrument Schedule or Definitions section "
                "to find its name (e.g. 'Discharge Pressure'). "
                "You never output 'Unnamed Component' if the definition exists anywhere in the text."
            ),
            llm=self.llm,
            verbose=True
        )

        task = Task(
            description=(
                "Analyze the COMPLETE document below to extract Control Logic.\n"
                "================================================================\n"
                f"{context_prefix}"
                f"{text}\n"
                "================================================================\n\n"
                "**INSTRUCTIONS:**\n"
                "1. **Identify Loops**: Group logic by equipment/system (e.g. 'Booster Pump Control').\n"
                "2. **Resolve Tags**: For every input/output tag, find its real name in the document. Do not use 'Unknown'.\n"
                "3. **Extract Interlocks**: explicit safety trips, start permissives, and auto-shutdowns.\n"
                "4. **Output**: Return a JSON object strictly following the LogicExtractionResult schema."
            ),
            expected_output="A complete LogicExtractionResult JSON.",
            agent=agent,
            output_json=LogicExtractionResult
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        return self._safe_extract(crew.kickoff())

    # --------------------------------------------------------------------------
    # Strategy B: Map-Reduce (Only for > 1M tokens)
    # --------------------------------------------------------------------------
    def _run_map_reduce(self, text: str, context_prefix: str):
        # Implementation of Map-Reduce for massive files (same logic as before, just larger chunks)
        # Note: In 99% of cases, you won't hit this with Gemini 2.5 Flash.
        chunks = self._split_text(text, self.CHUNK_SIZE_CHARS)
        print(f"✂️ Split document into {len(chunks)} chunks.")
        
        all_loops = []
        mapper = Agent(
            role='Logic Analyst',
            goal='Extract control loops from a text segment.',
            backstory="You analyze fragments. Report tags exactly as seen.",
            llm=self.llm,
            verbose=False
        )

        for i, chunk in enumerate(chunks):
            print(f"   Processing Chunk {i+1}/{len(chunks)}...")
            task = Task(
                description=f"Extract control logic from this fragment:\n\n{chunk}",
                expected_output="Partial LogicExtractionResult JSON.",
                agent=mapper,
                output_json=LogicExtractionResult
            )
            crew = Crew(agents=[mapper], tasks=[task], verbose=False)
            res = self._safe_extract(crew.kickoff())
            if res and 'loops' in res:
                all_loops.extend(res['loops'])
        
        # Merge
        reducer = Agent(
            role='Lead Engineer', 
            goal='Merge and Deduplicate Control Loops.', 
            backstory="You merge partial lists into a master list.", 
            llm=self.llm
        )
        task = Task(
            description=f"Merge these loops:\n{json.dumps(all_loops)[:500000]}...", # Truncate if insanely huge
            expected_output="Final LogicExtractionResult JSON.",
            agent=reducer,
            output_json=LogicExtractionResult
        )
        final_crew = Crew(agents=[reducer], tasks=[task], verbose=True)
        return self._safe_extract(final_crew.kickoff())

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _split_text(self, text, chunk_size, overlap=2000):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap 
        return chunks

    def _safe_extract(self, result):
        """Robust extraction for Pydantic/JSON/Raw output."""
        try:
            if hasattr(result, 'json_dict') and result.json_dict:
                return result.json_dict
            elif hasattr(result, 'pydantic') and result.pydantic:
                return result.pydantic.model_dump()
            elif hasattr(result, 'raw'):
                clean = result.raw.replace('```json', '').replace('```', '').strip()
                return json.loads(clean)
        except Exception as e:
            print(f"❌ Extraction Error: {e}")
        return {"loops": []}