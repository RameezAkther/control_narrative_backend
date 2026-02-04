import os
import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. Output Schema
# ==============================================================================

class ControlNarrativeSummary(BaseModel):
    """Structured summary focusing on Automation and Control logic."""
    system_description: Optional[str] = Field(
        default="No description extracted.", 
        description="A comprehensive description of the system boundary."
    )
    operational_modes: Optional[str] = Field(
        default="No modes extracted.", 
        description="Detailed description of operational modes."
    )
    major_control_functions: List[str] = Field(
        default_factory=list,
        description="List of specific control strategies, loops, or sequences."
    )
    safety_interlocks_overview: Optional[str] = Field(
        default="No safety logic extracted.", 
        description="Summary of critical safety interlocks."
    )
    primary_equipment_tags: List[str] = Field(
        default_factory=list,
        description="List of unique equipment tags found."
    )

# ==============================================================================
# 2. The Hybrid Agent Runner
# ==============================================================================

class UnderstandingAgentRunner:
    def __init__(self, model_name="gemini-2.5-flash"): 
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        # Threshold: ~800k tokens (approx 3.2M chars) to be safe
        self.MAX_SINGLE_SHOT_CHARS = 3_200_000 
        self.CHUNK_SIZE_CHARS = 100_000 # 100k chars per chunk for the "Big Path"

    def run(self, sections: List[Dict]) -> Dict[str, Any]:
        """Main entry point. Decides strategy based on size."""
        
        # 1. Flatten document
        full_text = self._construct_full_context(sections)
        doc_length = len(full_text)
        
        print(f"Document Loaded. Length: {doc_length:,} chars.")

        # 2. Decide Strategy
        if doc_length < self.MAX_SINGLE_SHOT_CHARS:
            print("Strategy: SINGLE-SHOT (Fits in context)")
            return self._run_single_shot(full_text)
        else:
            print("Strategy: MAP-REDUCE (Too large, chunking required)")
            return self._run_map_reduce(full_text)

    # --------------------------------------------------------------------------
    # Strategy A: Single Shot (Best Quality)
    # --------------------------------------------------------------------------
    def _run_single_shot(self, text: str):
        agent = Agent(
            role='Senior Control Systems Lead',
            goal='Analyze the FULL document and produce a unified Control Narrative.',
            backstory="You are an expert Automation Engineer. You see the entire document at once.",
            llm=self.llm,
            verbose=True
        )
        
        task = Task(
            description=f"Analyze the following COMPLETE document:\n\n{text}\n\nExtract the ControlNarrativeSummary.",
            expected_output="A complete ControlNarrativeSummary JSON.",
            agent=agent,
            output_json=ControlNarrativeSummary
        )
        
        crew = Crew(agents=[agent], tasks=[task], verbose=True)
        return self._safe_extract(crew.kickoff())

    # --------------------------------------------------------------------------
    # Strategy B: Map-Reduce (For Infinite Scale)
    # --------------------------------------------------------------------------
    def _run_map_reduce(self, text: str):
        # 1. SPLIT (Map)
        chunks = self._split_text(text, self.CHUNK_SIZE_CHARS)
        print(f"Split document into {len(chunks)} chunks.")
        
        partial_results = []
        
        # Define the 'Map' Agent
        mapper = Agent(
            role='Section Analyst',
            goal='Extract control details strictly from the provided text fragment.',
            backstory="You analyze one chapter of a massive spec. You only report what you see.",
            llm=self.llm,
            verbose=False # Keep quiet
        )

        for i, chunk in enumerate(chunks):
            print(f"   Processing Chunk {i+1}/{len(chunks)}...")
            task = Task(
                description=(
                    f"Analyze this PARTIAL text fragment ({i+1}/{len(chunks)}):\n\n{chunk}\n\n"
                    "Extract any tags, logic, or descriptions found HERE."
                ),
                expected_output="A partial ControlNarrativeSummary JSON.",
                agent=mapper,
                output_json=ControlNarrativeSummary
            )
            crew = Crew(agents=[mapper], tasks=[task], verbose=False)
            res = self._safe_extract(crew.kickoff())
            if res:
                partial_results.append(res)
        
        # 2. MERGE (Reduce)
        print(f"Merging {len(partial_results)} partial summaries...")
        
        # Dump partials to string for the reducer
        partials_str = json.dumps(partial_results, indent=2)
        
        reducer = Agent(
            role='Chief Engineer',
            goal='Merge multiple partial reports into one Master Control Narrative.',
            backstory="You take fragment reports, remove duplicates (e.g. merge P-101 findings), and write the final spec.",
            llm=self.llm,
            verbose=True
        )
        
        reduce_task = Task(
            description=(
                f"Here are {len(partial_results)} partial analysis reports from a massive document:\n"
                f"{partials_str}\n\n"
                "**YOUR JOB:**\n"
                "1. Consolidate all Equipment Tags (Remove duplicates).\n"
                "2. Merge Control Functions into a single list.\n"
                "3. Synthesize the System Description from the partial notes.\n"
                "4. Return the FINAL merged JSON."
            ),
            expected_output="The final merged ControlNarrativeSummary JSON.",
            agent=reducer,
            output_json=ControlNarrativeSummary
        )
        
        final_crew = Crew(agents=[reducer], tasks=[reduce_task], verbose=True)
        return self._safe_extract(final_crew.kickoff())

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------
    def _construct_full_context(self, sections: List[Dict]) -> str:
        return "\n".join([f"# {s.get('title','')}\n{s.get('content','')}" for s in sections])

    def _split_text(self, text, chunk_size, overlap=2000):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap # Overlap for context safety
        return chunks

    def _safe_extract(self, result):
        if hasattr(result, 'json_dict') and result.json_dict:
            return result.json_dict
        elif hasattr(result, 'pydantic') and result.pydantic:
            return result.pydantic.dict()
        try:
            return json.loads(result.raw.replace('```json','').replace('```',''))
        except:
            return {}
        
# ==============================================================================
# 3. Helper: Parse Markdown File (Unchanged)
# ==============================================================================

def parse_markdown_file(file_path: str) -> List[Dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sections = []
    current_title = "Document Start"
    current_content = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_content:
                sections.append({
                    "title": current_title,
                    "content": "\n".join(current_content).strip()
                })
            current_title = stripped.lstrip('#').strip()
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_content).strip()
    })

    return sections
