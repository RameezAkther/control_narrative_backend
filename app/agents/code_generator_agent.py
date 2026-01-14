import os
import json
import re
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# The Agent Runner (Modular & Batched)
# ==============================================================================

class CodeGeneratorRunner:
    def __init__(self, model_name="gemini-2.0-flash-exp"):
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
        
        # 1. Extract Valid Loops Only
        # We shouldn't generate code for broken loops.
        # This is a basic filter; in production you might want to be more specific.
        raw_loops = logic_data.get('loops', [])
        if not raw_loops:
            # Fallback if logic_data structure is different (e.g. if passed merged data)
            raw_loops = logic_data if isinstance(logic_data, list) else []

        print(f"🏭 Code Gen: Preparing to write code for {len(raw_loops)} loops...")

        # 2. Define Agents
        
        # Agent A: The Architect (Defines Variables)
        # This prevents "Variable not defined" errors by doing it all upfront.
        architect = Agent(
            role='PLC Architect',
            goal='Define all VAR_INPUT, VAR_OUTPUT, and internal VARs.',
            backstory="You are a strict compiler. You declare every single tag found in the requirements.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # Agent B: The Developer (Writes Logic)
        developer = Agent(
            role='Senior PLC Developer',
            goal='Write efficient Structured Text (ST) logic for specific loops.',
            backstory="You write clean IF/THEN/ELSE and PID function blocks. You use the exact tag names provided.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # 3. Create Tasks
        tasks = []

        # --- TASK 1: Variable Declaration (The "Header") ---
        # We send ALL loops to the architect so they can declare everything global/local.
        # If the list is huge, this might need chunking too, but variable lists are usually denser/smaller than logic.
        header_task = Task(
            description=(
                f"Analyze these Control Loops:\n{json.dumps(raw_loops[:30], indent=2)} "
                f"\n(Truncated list for brevity if huge)...\n\n"
                "TASK:\n"
                "1. Extract ALL unique Tag Names (inputs and outputs).\n"
                "2. Generate the 'VAR', 'VAR_INPUT', and 'VAR_OUTPUT' blocks.\n"
                "3. Assume standard data types (BOOL for switches, REAL for transmitters).\n"
                "4. Return ONLY the variable declaration text."
            ),
            expected_output="IEC 61131-3 Variable Declaration Block.",
            agent=architect
        )
        tasks.append(header_task)

        # --- TASK 2..N: Logic Generation (The "Body") ---
        # We batch the logic writing to ensure high quality code for every loop.
        batches = self._batch_loops(raw_loops, batch_size=5)
        
        for i, batch in enumerate(batches):
            task = Task(
                description=(
                    f"Write Structured Text (ST) logic for this BATCH of loops (Batch {i+1}/{len(batches)}):\n\n"
                    f"{json.dumps(batch, indent=2)}\n\n"
                    "RULES:\n"
                    "1. Use standard IEC 61131-3 syntax.\n"
                    "2. Handle interlocks: IF (Interlock_Condition) THEN Output := FALSE; END_IF;\n"
                    "3. Return ONLY the code logic. Do NOT repeat VAR declarations."
                ),
                expected_output="Structured Text Logic Snippet.",
                agent=developer
            )
            tasks.append(task)

        # 4. Execute
        crew = Crew(
            agents=[architect, developer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()

        # 5. Assemble The Final File
        # CrewAI returns the final task's output by default, but we need ALL outputs.
        # We can access them via the task objects if we kept references, or we can use a custom approach.
        # For simplicity in this script, we will assume we need to concatenate manually or rely on a final "Merger" agent.
        
        # Let's add a final "Merger" agent to be safe and let the AI handle the stitching.
        merger = Agent(
            role='Code Integrator',
            goal='Combine variable declarations and logic chunks into one file.',
            backstory="You take the headers and the body code and paste them into a final PROGRAM block.",
            llm=self.llm,
            verbose=True
        )

        merge_task = Task(
            description="Combine the Variable Declarations and all Logic Snippets from previous tasks into one final valid ST PROGRAM.",
            expected_output="Final Complete PLC Program Code.",
            agent=merger,
            context=tasks # Gives access to all previous outputs
        )

        final_crew = Crew(
            agents=[architect, developer, merger],
            tasks=[*tasks, merge_task],
            process=Process.sequential,
            verbose=True
        )

        final_result = final_crew.kickoff()

        raw_text = ""
        if hasattr(final_result, 'raw'):
            raw_text = final_result.raw
        else:
            raw_text = str(final_result)
            
        return self.clean_code_output(raw_text)