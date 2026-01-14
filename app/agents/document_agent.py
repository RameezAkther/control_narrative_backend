"""
DocumentAgent

Reworked to support two deterministic parsing strategies:
- fast: deterministic parser (no LLM, quick)
- accurate: docling-based parser (slower, more accurate)

Both parsers are implemented in `app.agents.parsers` and the agent simply dispatches
based on the `parsing_strategy` argument provided by the frontend.
"""

import os
from typing import List, Dict

from app.agents.parsers import DeterministicPDFParser, DoclingPDFParser

class DocumentAgent:
    def __init__(self, *args, **kwargs):
        pass

    def _deterministic_to_sections(self, parse_result: Dict) -> List[Dict]:
        """
        Convert deterministic pages->sections.
        Simple heuristic: one section per page with page content concatenated in reading order.
        """
        sections = []
        pages = parse_result.get("pages", [])

        for p in pages:
            title = f"Page {p.get('page_number') }"
            parts = []
            for block in p.get("blocks", []):
                # preserve placeholders and text
                parts.append(block.get("content", ""))
            content = "\n\n".join(parts).strip()
            sections.append({"title": title, "content": content})

        return sections

    def _docling_to_sections(self, parse_result: Dict) -> List[Dict]:
        # Docling parser already returns a `sections` list from markdown headings
        return parse_result.get("sections", [])

    def run(self, file_path: str, parsing_strategy: str = "fast", output_dir: str = None) -> Dict:
        """
        Run document parsing.

        Args:
            file_path: path to PDF file
            parsing_strategy: 'fast' -> deterministic, 'accurate' -> docling
            output_dir: optional directory where parser should write images/markdown
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        if parsing_strategy not in ("fast", "accurate"):
            raise ValueError("parsing_strategy must be 'fast' or 'accurate'")

        if parsing_strategy == "fast":
            print("Using DeterministicPDFParser for fast parsing...")
            print(f"Output dir: {output_dir}")
            print(f"File path: {file_path}")
            parser = DeterministicPDFParser(file_path, output_dir=output_dir)
            print("Parsing document...")
            result = parser.parse()
            print("Parsing complete.")
            sections = self._deterministic_to_sections(result)
            print("Converted to sections.")
            images = result.get("images", {})
            metadata = result.get("metadata", {})
            metadata.update({"strategy": "fast"})

        else:  # accurate
            # Docling parser may not be installed; provide helpful error
            if DoclingPDFParser is None:
                raise RuntimeError("Docling parser not available. Install 'docling' and 'docling-core' to enable accurate parsing.")
            print("Using DoclingPDFParser for accurate parsing...")
            parser = DoclingPDFParser(file_path, output_dir=output_dir)
            result = parser.parse()
            sections = self._docling_to_sections(result)
            images = result.get("images", {})
            metadata = result.get("metadata", {})
            metadata.update({"strategy": "accurate"})

        # Propagate markdown_path (if any) and parsed folder
        markdown_path = result.get("markdown_path") or result.get("markdown")
        if markdown_path:
            metadata["markdown_path"] = markdown_path

        if output_dir:
            metadata.setdefault("parsed_folder", output_dir)

        return {
            "sections": sections,
            "images": images,
            "metadata": metadata,
        }
