import os
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
# 2. The Agent Runner (Map-Reduce Version)
# ==============================================================================

class UnderstandingAgentRunner:
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
            # Format the section text
            sec_text = f"\n--- SECTION: {sec.get('title', 'Unknown')} ---\n"
            sec_text += sec.get('content', '') + "\n"
            
            # If adding this section exceeds max size, push current chunk and start new
            if len(current_chunk) + len(sec_text) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sec_text
            else:
                current_chunk += sec_text
        
        # Add the last remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def run(self, sections: List[Dict]):
        # 1. Chunk the document
        doc_chunks = self._create_chunks(sections)
        print(f"📄 Document split into {len(doc_chunks)} chunks for processing.")

        # --- AGENTS ---
        
        # The Mapper: Analyzes specific parts of the document
        section_analyst = Agent(
            role='Section Control Analyst',
            goal='Extract control details, tags, and logic from a specific document section.',
            backstory=(
                "You are an expert at reading FDS fragments. You do not guess. "
                "You only report what is explicitly present in the text provided to you."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

        # The Reducer: Merges everything
        lead_engineer = Agent(
            role='Lead Control Systems Engineer',
            goal='Synthesize multiple partial reports into one final Control Narrative.',
            backstory=(
                "You receive reports from various analysts about different parts of a system. "
                "Your job is to de-duplicate equipment tags, merge system descriptions into a coherent summary, "
                "and consolidate safety logic."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

        # --- TASKS ---

        map_tasks = []
        
        # MAP STEP: Create a task for each chunk
        for i, chunk_text in enumerate(doc_chunks):
            task = Task(
                description=(
                    f"Analyze this PARTIAL text (Chunk {i+1}/{len(doc_chunks)}):\n\n"
                    f"{chunk_text}\n\n"
                    "Extract the following ONLY if found in this text:\n"
                    "1. Equipment Tags (e.g. P-101)\n"
                    "2. Control Modes mentioned here\n"
                    "3. Any System Description details found here\n"
                    "Return a JSON summary of THIS chunk."
                ),
                expected_output="A partial JSON summary of this specific text chunk.",
                agent=section_analyst,
                output_json=ControlNarrativeSummary # We use the same schema for intermediate steps
            )
            map_tasks.append(task)

        # REDUCE STEP: One final task that depends on all map tasks
        reduce_task = Task(
            description=(
                "You have received multiple partial summaries from the document analysis. "
                "Your task is to MERGE them into one final Master Report.\n\n"
                "1. **Consolidate Descriptions**: Combine the system descriptions into one coherent overview.\n"
                "2. **Merge Tags**: Combine all 'primary_equipment_tags' lists and REMOVE duplicates.\n"
                "3. **Merge Functions**: Combine control functions and modes.\n"
                "4. **Final Output**: Return the final consolidated JSON."
            ),
            expected_output="The final merged ControlNarrativeSummary JSON.",
            agent=lead_engineer,
            context=map_tasks, # <--- This passes the output of all map tasks to this task
            output_json=ControlNarrativeSummary
        )

        # --- CREW EXECUTION ---
        
        crew = Crew(
            agents=[section_analyst, lead_engineer],
            tasks=[*map_tasks, reduce_task], # Run maps first, then reduce
            process=Process.sequential,
            verbose=True
        )

        return crew.kickoff()

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