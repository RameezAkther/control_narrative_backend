import os
import re
import ollama
from pathlib import Path

class RobustImageAnalyzer:
    def __init__(self, model_name="qwen3-vl"):
        self.model = model_name

    def analyze(self, image_path, context_text="", specific_type=None):
        """
        Routes the image to the correct prompting strategy.
        specific_type: 'TABLE', 'VISUAL', or None (Auto-detect for Type 2)
        """
        if not os.path.exists(image_path):
            return "[ERROR: Image file not found]"

        # 1. TABLE HANDLING
        if specific_type == "TABLE":
            return self._process_table(image_path)
        
        # 2. VISUAL/DIAGRAM HANDLING
        elif specific_type == "VISUAL":
            return self._process_diagram(image_path, context_text)
        
        # 3. AUTO-DETECT (For Type 2 where we don't know the type beforehand)
        else:
            return self._process_autodetect(image_path, context_text)

    def _process_table(self, image_path):
        prompt = """
        Analyze this image.
        1. If it is a valid data table, extract all values and return them strictly as a Markdown Table.
        2. If it is NOT a table (e.g., a chart, a blurry line, or a blank box), return exactly the string: "NO_CONTENT".
        Do not add conversational text. Just the markdown table or NO_CONTENT.
        """
        return self._query_ollama(prompt, image_path)

    def _process_diagram(self, image_path, context_text):
        prompt = f"""
        You are a technical document assistant. Analyze this image.
        
        CONTEXT FROM DOCUMENT:
        "{context_text}"
        
        INSTRUCTIONS:
        1. NOISE CHECK: If this is a company logo, a decorative line, or a generic stock photo, return exactly: "NO_CONTENT".
        2. DIAGRAMS: If it is a P&ID, Flow Chart, or Control Panel, describe the visual cues (buttons, lights, flow direction, labels).
        3. INTEGRATION: Relate the image elements to the provided context if possible.
        
        Return the description in plain text or markdown.
        """
        return self._query_ollama(prompt, image_path)

    def _process_autodetect(self, image_path, context_text):
        # Combined prompt for Type 2 where tags are generic
        prompt = f"""
        Analyze this image found in a technical manual.
        
        CONTEXT: "{context_text}"
        
        TASK:
        - If it is a **Table**: Convert it to a Markdown Table.
        - If it is a **Diagram/Panel**: Describe the equipment, states, and labels visible.
        - If it is **Noise** (logo/header/footer artifact): Return exactly "NO_CONTENT".
        """
        return self._query_ollama(prompt, image_path)

    def _query_ollama(self, prompt, image_path):
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path]
                }]
            )
            content = response['message']['content'].strip()
            if "NO_CONTENT" in content:
                return "" # Return empty string to remove placeholder effectively
            return f"\n\n> **Image Analysis:**\n{content}\n\n"
        except Exception as e:
            return f"\n[System Error Processing Image: {e}]\n"


class MarkdownEnricher:
    def __init__(self, base_dir, model_name="qwen3-vl"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.analyzer = RobustImageAnalyzer(model_name)

    def get_context(self, full_text, start_index, end_index, window=500):
        """Extracts text around the match for context."""
        prev_text = full_text[max(0, start_index - window):start_index]
        next_text = full_text[end_index:min(len(full_text), end_index + window)]
        return (prev_text + " ... " + next_text).replace("\n", " ")

    def process_type_1(self, content):
        """
        Handles explicit placeholders: 
        [Visual Context: ... img_001] or [Table Placeholder: ... img_002]
        """
        # Regex to capture: 1. Type, 2. Description (optional), 3. Filename
        # Matches: [Visual Context: some text img_001]
        pattern = re.compile(r'\[(Visual Context|Table Placeholder):.*?\b(img_\d+|image_\d+).*?\]', re.IGNORECASE)
        
        def replacer(match):
            full_match = match.group(0)
            tag_type = match.group(1) # "Visual Context" or "Table Placeholder"
            
            # Extract filename more robustly from the tag
            # Looking for patterns like img_001 or image_01 inside the brackets
            file_match = re.search(r'(img_\d+|image_\d+)', full_match, re.IGNORECASE)
            if not file_match:
                return full_match # Keep original if filename parse fails
            
            filename = file_match.group(1)
            
            # Try extensions (tif, png, jpg) - Assuming standard
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                p = self.images_dir / f"{filename}{ext}"
                if p.exists():
                    image_path = str(p)
                    break
            
            if not image_path:
                return f"\n*[Missing Image File: {filename}]*\n"

            # Context extraction
            context = ""
            analysis_type = "TABLE" if "Table" in tag_type else "VISUAL"
            
            if analysis_type == "VISUAL":
                context = self.get_context(content, match.start(), match.end())

            print(f"Processing Type 1: {filename} as {analysis_type}...")
            return self.analyzer.analyze(image_path, context, analysis_type)

        return pattern.sub(replacer, content)

    def process_type_2(self, content):
        """
        Handles generic placeholders: Assumes sequential images: picture-01.png, picture-02.png...
        """
        pattern = re.compile(r'', re.IGNORECASE)
        
        # We need to manually iterate to keep track of the index
        parts = []
        last_pos = 0
        img_counter = 1
        
        for match in pattern.finditer(content):
            # Append text before the image
            parts.append(content[last_pos:match.start()])
            
            # Calculate Context
            context = self.get_context(content, match.start(), match.end())
            
            # construct filename: picture-01.png, picture-02.png (pad with zero if needed)
            # You might need to adjust formatting based on your exact file naming (e.g., picture-1 vs picture-01)
            filename = f"picture-{img_counter:02d}.png" 
            image_path = self.images_dir / filename
            
            # If 02d doesn't exist, try 01d or just number
            if not image_path.exists():
                 image_path = self.images_dir / f"picture-{img_counter}.png"

            print(f"Processing Type 2: Index {img_counter} ({filename})...")
            
            description = self.analyzer.analyze(str(image_path), context, specific_type=None)
            parts.append(description)
            
            last_pos = match.end()
            img_counter += 1
            
        parts.append(content[last_pos:])
        return "".join(parts)

# ==========================================
# EXECUTION
# ==========================================

def run_image_parser(input_md_path, parser_type=1):
    print(f"Enriching Markdown: {input_md_path} (Type {parser_type})")
    base_dir = os.path.dirname(input_md_path)
    with open(input_md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    enricher = MarkdownEnricher(base_dir, model_name="qwen3-vl")

    if parser_type == 1:
        new_content = enricher.process_type_1(content)
    elif parser_type == 2:
        new_content = enricher.process_type_2(content)
    else:
        print("Invalid Type")
        return

    output_path = input_md_path.replace(".md", "_enriched.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Done! Saved to {output_path}")
