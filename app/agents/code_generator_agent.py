import os
import json
import re
import time
from typing import List, Dict, Union
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

# ==============================================================================
# The Code Generator Runner (Single-Shot Efficiency)
# ==============================================================================

class CodeGeneratorRunner:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            verbose=True,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.MAX_LOOPS_SINGLE_SHOT = 2000

    def clean_code_output(self, text: str) -> str:
        """Strips Markdown and extra chatter to leave raw PLC code."""
        # 1. Remove Markdown code blocks (```st, ```iecst, ```)
        text = re.sub(r'```[a-zA-Z]*', '', str(text))
        text = re.sub(r'```', '', text)
        
        # 2. Trim whitespace
        text = text.strip()
        
        # 3. Ensure it looks like code (simple heuristic)
        if "PROGRAM" not in text and "VAR" not in text:
            text = f"(* Generated Code Fragment *)\n{text}"
            
        return text

    def run(self, logic_data: Union[Dict, List], validation_report: Dict) -> str:
        """
        Generates the full IEC 61131-3 Structured Text program.
        Args:
            logic_data: The list of control loops (sensors, actuators, interlocks).
            validation_report: The report containing strategy types and criticality.
        """
        
        # 1. Normalize Input
        if isinstance(logic_data, dict):
            raw_loops = logic_data.get('loops', [])
        else:
            raw_loops = logic_data
            
        if not raw_loops:
            return "(* No Control Loops provided to generate code. *)"

        # 2. Enrich Loops with Strategy Info (from Validation Report if available)
        # We try to map the specific strategy (PID vs Interlock) to the loop
        enriched_loops = []
        issues_map = {i.get('loop_name'): i for i in validation_report.get('issues', [])}
        
        # If the validation report contains the full context, use that instead
        # (This depends on how you pass data between agents. We assume raw_loops is the source of truth).
        
        print(f"Code Gen: Generating Full PLC Program for {len(raw_loops)} loops...")

        # 3. Check Size
        if len(raw_loops) > self.MAX_LOOPS_SINGLE_SHOT:
            print(f"Truncating to {self.MAX_LOOPS_SINGLE_SHOT} loops.")
            raw_loops = raw_loops[:self.MAX_LOOPS_SINGLE_SHOT]

        # 4. Define Agents
        # We use one "Lead Developer" to ensure the VARs and Logic are consistent.
        lead_dev = Agent(
            role='Senior PLC Developer',
            goal='Write a complete, compilable IEC 61131-3 Structured Text (ST) program.',
            backstory=(
                "You are an expert in CODESYS and TIA Portal. "
                "You MUST declare every variable used. "
                "You write defensive code: Interlocks always override Control logic. "
                "You use standard Hungarian Notation (e.g., xStart, rLevel, iState)."
            ),
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # 5. Prepare Context
        context_str = json.dumps(raw_loops, indent=2)

        task = Task(
            description=(
                "Generate a COMPLETE IEC 61131-3 Structured Text program for the following system:\n"
                "================================================================\n"
                f"{context_str}\n"
                "================================================================\n\n"
                "**REQUIREMENTS:**\n"
                "1. **Structure**: Wrap the code in `PROGRAM MainControl ... END_PROGRAM`.\n"
                "2. **Variables**: Generate a `VAR` block declaring ALL sensors, actuators, and internal states. Use `BOOL` for switches/valves and `REAL` for transmitters.\n"
                "3. **Logic**: Write the logic for each loop.\n"
                "   - If it's a **Motor/Pump**: Implement Start/Stop logic with Interlocks overriding the Start command.\n"
                "   - If it's a **Valve**: Implement Open/Close logic.\n"
                "   - If it's a **PID**: Call a hypothetical `FB_PID` block (do not write the PID internals, just the call).\n"
                "4. **Comments**: Add comments explaining which loop the code belongs to.\n"
                "5. **Output**: Return ONLY the raw code. No markdown formatting."
            ),
            expected_output="Raw IEC 61131-3 Structured Text code.",
            agent=lead_dev
        )

        # 6. Execute
        crew = Crew(
            agents=[lead_dev],
            tasks=[task],
            verbose=True
        )

        try:
            print("Generating Code...")
            start_time = time.time()
            result = crew.kickoff()
            elapsed = time.time() - start_time
            print(f"Code generation complete in {elapsed:.2f} seconds.")

            # Robust Extraction
            raw_output = ""
            if hasattr(result, 'raw'):
                raw_output = result.raw
            else:
                raw_output = str(result)
            
            cleaned_code = self.clean_code_output(raw_output)
            return cleaned_code

        except Exception as e:
            print(f"Code Gen Failed: {e}")
            return f"(* Error generating code: {e} *)"

# ==============================================================================
# Usage Example
# ==============================================================================
# if __name__ == "__main__":
#     mock_loops = [
#         {"loop_name": "P-101", "inputs": [{"tag": "LIT-101", "role": "Sensor"}], "outputs": [{"tag": "P-101-CMD", "role": "Actuator"}]}
#     ]
#     runner = CodeGeneratorRunner()
#     code = runner.run(mock_loops, {})
#     print(code)