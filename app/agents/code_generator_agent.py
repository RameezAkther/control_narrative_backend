import os
import json
import re
import time
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# The Agent Runner (Modular & Batched)
# ==============================================================================

class CodeGeneratorRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def clean_code_output(self, text: str) -> str:
        """Strips Markdown and extra chatter."""
        # Remove ```iecst, ```st, or just ``` lines
        text = re.sub(r'```[a-zA-Z]*', '', str(text))
        text = re.sub(r'```', '', text)
        return text.strip()

    def _batch_loops(self, loops: List[Dict], batch_size=5) -> List[List[Dict]]:
        return [loops[i:i + batch_size] for i in range(0, len(loops), batch_size)]

    def run(self, logic_data: Dict, validation_report: Dict) -> str:
        """
        Args:
            logic_data: The dictionary output from ValidatorRunner (or LogicAgent).
            validation_report: The dictionary output from ValidatorRunner.
        """
        
        # 1. Extract Valid Loops
        raw_loops = logic_data.get('loops', [])
        if not raw_loops:
            # Fallback
            raw_loops = logic_data if isinstance(logic_data, list) else []

        print(f"🏭 Code Gen: Preparing to write code for {len(raw_loops)} loops...")

        # 2. Define Agents
        architect = Agent(
            role='PLC Architect',
            goal='Define all VAR_INPUT, VAR_OUTPUT, and internal VARs.',
            backstory="You are a strict compiler. You declare every single tag found in the requirements.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        developer = Agent(
            role='Senior PLC Developer',
            goal='Write efficient Structured Text (ST) logic for specific loops.',
            backstory="You write clean IF/THEN/ELSE and PID function blocks.",
            llm=self.llm,
            verbose=False, # Reduce noise for batching
            allow_delegation=False
        )

        # --- PHASE 1: HEADER GENERATION ---
        print("\n📝 Generating PLC Header (Variables)...")
        
        # We send a truncated list if huge to avoid context limits for the header prompt
        # Ideally, you'd want to extract just the tags first, but for now we send the structure
        header_task = Task(
            description=(
                f"Analyze these Control Loops:\n{json.dumps(raw_loops[:50], indent=2)} "
                f"\n(List truncated if > 50)...\n\n"
                "TASK:\n"
                "1. Extract ALL unique Tag Names (inputs and outputs) from the provided JSON.\n"
                "2. Generate the 'VAR', 'VAR_INPUT', and 'VAR_OUTPUT' blocks.\n"
                "3. Assume standard data types (BOOL for switches, REAL for transmitters).\n"
                "4. Return ONLY the variable declaration text."
            ),
            expected_output="IEC 61131-3 Variable Declaration Block.",
            agent=architect
        )

        header_crew = Crew(
            agents=[architect],
            tasks=[header_task],
            verbose=True
        )
        
        header_result = header_crew.kickoff()
        
        final_code_parts = []
        if hasattr(header_result, 'raw'):
            final_code_parts.append(self.clean_code_output(header_result.raw))
        else:
            final_code_parts.append(self.clean_code_output(str(header_result)))

        # --- PHASE 2: LOGIC GENERATION (Batched) ---
        print("\n⚙️ Generating Logic Blocks...")
        
        batches = self._batch_loops(raw_loops, batch_size=5)
        
        for i, batch in enumerate(batches):
            print(f"   Writing Logic for Batch {i+1}/{len(batches)}...")
            
            task = Task(
                description=(
                    f"Write Structured Text (ST) logic for this BATCH of loops:\n\n"
                    f"{json.dumps(batch, indent=2)}\n\n"
                    "RULES:\n"
                    "1. Use standard IEC 61131-3 syntax.\n"
                    "2. Handle interlocks: IF (Interlock_Condition) THEN Output := FALSE; END_IF;\n"
                    "3. Return ONLY the code logic. Do NOT repeat VAR declarations."
                ),
                expected_output="Structured Text Logic Snippet.",
                agent=developer
            )

            # Isolate Task
            crew = Crew(
                agents=[developer],
                tasks=[task],
                verbose=False
            )

            try:
                result = crew.kickoff()
                output_text = ""
                if hasattr(result, 'raw'):
                    output_text = result.raw
                else:
                    output_text = str(result)
                
                cleaned = self.clean_code_output(output_text)
                final_code_parts.append(f"\n(* --- Batch {i+1} Logic --- *)\n{cleaned}")
                print(f"   ✅ Batch {i+1} written.")

            except Exception as e:
                print(f"   ❌ Error writing Batch {i+1}: {e}")
                final_code_parts.append(f"\n(* Error generating Batch {i+1} *)")

            # Rate Limiting
            time.sleep(2)

        # --- FINAL ASSEMBLY ---
        full_program = "\n\n".join(final_code_parts)
        
        # Wrap in PROGRAM block if missing
        if "PROGRAM" not in full_program:
            full_program = f"PROGRAM MainControl\n{full_program}\nEND_PROGRAM"

        return full_program