import os
import json
import time
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# 1. Output Schema
# ==============================================================================

class ControlNarrativeSummary(BaseModel):
    """Structured summary focusing on Automation and Control logic."""
    system_description: Optional[str] = Field(
        default="System description could not be extracted.",
        description="Brief description of the system boundary."
    )
    operational_modes: Optional[str] = Field(
        default="No specific operational modes identified.",
        description="Identified modes of operation (e.g., Auto, Manual)."
    )
    major_control_functions: List[str] = Field(
        default_factory=list,
        description="List of primary control loops or sequences."
    )
    safety_interlocks_overview: Optional[str] = Field(
        default="No high-level safety logic found.",
        description="High-level notes on safety logic."
    )
    primary_equipment_tags: List[str] = Field(
        default_factory=list,
        description="List of equipment tags found."
    )

# ==============================================================================
# 2. The Agent Runner (Robust Batching Version)
# ==============================================================================

class UnderstandingAgentRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def _create_chunks(self, sections: List[Dict], max_chunk_size=20000) -> List[str]:
        """
        Groups sections into chunks. 
        Lowered to 20k to ensure we don't hit output token limits on summaries.
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

    def run(self, sections: List[Dict]):
        # 1. Chunk the document
        doc_chunks = self._create_chunks(sections)
        print(f"📄 Understanding Agent: Document split into {len(doc_chunks)} chunks.")

        # --- AGENTS ---
        
        section_analyst = Agent(
            role='Section Control Analyst',
            goal='Extract control details, tags, and logic from a specific document section.',
            backstory="You are an expert at reading FDS fragments. You only report what is explicitly present in the text provided.",
            llm=self.llm,
            allow_delegation=False,
            verbose=False # Keep it quiet during batching
        )

        lead_engineer = Agent(
            role='Lead Control Systems Engineer',
            goal='Synthesize multiple partial reports into one final Control Narrative.',
            backstory="You merge partial reports, de-duplicate tags, and create a coherent system summary.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

        # --- PHASE 1: MAP (Process Chunks Sequentially) ---
        
        partial_summaries = []
        
        print("\n🔄 Starting Batch Summarization (Map Phase)...")

        for i, chunk_text in enumerate(doc_chunks):
            print(f"   Processing Chunk {i+1}/{len(doc_chunks)}...")

            task = Task(
                description=(
                    f"Analyze this PARTIAL text (Chunk {i+1}/{len(doc_chunks)}):\n\n"
                    f"{chunk_text}\n\n"
                    "Extract the following ONLY if found in this text:\n"
                    "1. Equipment Tags (e.g. P-101)\n"
                    "2. Control Modes mentioned here\n"
                    "3. Any System Description details found here\n"
                    "4. Major Control Functions mentioned here\n"
                    "Return a JSON summary of THIS chunk."
                ),
                expected_output="A partial JSON summary of this specific text chunk.",
                agent=section_analyst,
                output_json=ControlNarrativeSummary
            )

            # Isolate task in its own crew
            crew = Crew(
                agents=[section_analyst],
                tasks=[task],
                verbose=False
            )

            try:
                result = crew.kickoff()
                
                # Robust Result Extraction
                chunk_data = None
                if hasattr(result, 'json_dict') and result.json_dict:
                    chunk_data = result.json_dict
                elif hasattr(result, 'pydantic') and result.pydantic:
                    chunk_data = result.pydantic.dict()
                elif hasattr(result, 'raw'):
                    try:
                        clean = result.raw.replace('```json', '').replace('```', '')
                        chunk_data = json.loads(clean)
                    except:
                        pass
                
                if chunk_data:
                    partial_summaries.append(chunk_data)
                    print(f"   ✅ Chunk {i+1} summarized.")
                else:
                    print(f"   ⚠️ Chunk {i+1} produced no valid data.")

            except Exception as e:
                print(f"   ❌ Error processing Chunk {i+1}: {e}")

            # RATE LIMITING: Sleep between chunks
            time.sleep(2)

        # --- PHASE 2: REDUCE (Merge Results) ---
        
        print(f"\n🧩 Starting Aggregation (Reduce Phase) with {len(partial_summaries)} partial summaries...")

        if not partial_summaries:
            return {}

        # Prepare context for the reducer
        # We assume the list of summaries fits in the context window (Gemini 2.0 has 1M token window, so this is fine)
        context_str = json.dumps(partial_summaries, indent=2)

        reduce_task = Task(
            description=(
                "You have received multiple partial summaries from the document analysis. "
                "Your task is to MERGE them into one final Master Report.\n\n"
                f"RAW PARTIAL DATA:\n{context_str}\n\n"
                "INSTRUCTIONS:\n"
                "1. **Consolidate Descriptions**: Combine the system descriptions into one coherent overview.\n"
                "2. **Merge Tags**: Combine all 'primary_equipment_tags' lists and REMOVE duplicates.\n"
                "3. **Merge Functions**: Combine control functions and modes.\n"
                "4. **Final Output**: Return the final consolidated JSON."
            ),
            expected_output="The final merged ControlNarrativeSummary JSON.",
            agent=lead_engineer,
            output_json=ControlNarrativeSummary
        )

        final_crew = Crew(
            agents=[lead_engineer],
            tasks=[reduce_task],
            verbose=True
        )

        final_result = final_crew.kickoff()

        # Return standardized Dict
        if hasattr(final_result, 'json_dict') and final_result.json_dict:
            return final_result.json_dict
        elif hasattr(final_result, 'pydantic') and final_result.pydantic:
            return final_result.pydantic.dict()
        
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