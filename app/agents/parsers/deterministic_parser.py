import fitz  # PyMuPDF
import pdfplumber
import os
import re
import hashlib
from PIL import Image
from io import BytesIO
import tempfile

class DeterministicPDFParser:
    """
    Deterministic PDF parser focused on speed. Extracts text blocks, detects tables and visuals
    and writes images to an `output_dir/images` folder. Returns structured pages and images map.
    """

    def __init__(self, pdf_path, output_dir=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="parsed_")
        self.images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)

        self.image_counter = 0
        self.images = {}
        self.unique_image_hashes = {}

    def _clean_text(self, text):
        if not text:
            return None
        text = re.sub(r"[\._-]{3,}", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 3:
            return None
        return text

    def _save_image(self, image_bytes, page_num):
        image_hash = hashlib.md5(image_bytes).hexdigest()

        if image_hash in self.unique_image_hashes:
            return self.unique_image_hashes[image_hash]

        self.image_counter += 1
        image_id = f"img_{self.image_counter:03d}"
        image_path = os.path.join(self.images_dir, f"{image_id}.png")

        try:
            img = Image.open(BytesIO(image_bytes))
            img.save(image_path)
            self.images[image_id] = {"path": image_path, "first_seen_page": page_num}
            self.unique_image_hashes[image_hash] = image_id
            return image_id
        except Exception as e:
            print(f"Failed to save image: {e}")
            return None

    def _is_valid_table(self, table_obj):
        width = table_obj.bbox[2] - table_obj.bbox[0]
        height = table_obj.bbox[3] - table_obj.bbox[1]

        if width < 50 or height < 20:
            return False

        extracted_data = table_obj.extract()
        if not extracted_data:
            return False

        num_rows = len(extracted_data)
        num_cols = len(extracted_data[0]) if num_rows > 0 else 0

        if num_rows < 2 and num_cols < 2:
            return False

        return True

    def _process_tables_as_images(self, page_idx, fitz_page, plumber_page):
        """Extracts tables as images using the passed-in plumber page object."""
        table_blocks = []
        table_bboxes = []

        # Use the pre-opened plumber page
        tables = plumber_page.find_tables()

        for table in tables:
            if not self._is_valid_table(table):
                continue

            rect = fitz.Rect(table.bbox)

            pix = fitz_page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
            img_bytes = pix.tobytes("png")

            image_id = self._save_image(img_bytes, page_idx + 1)

            if image_id:
                table_blocks.append({
                    "type": "table_placeholder",
                    "content": f"[Table Placeholder: Table detected. See image {image_id}]",
                    "image_id": image_id,
                    "top": rect.y0,
                })
                table_bboxes.append(rect)

        return table_blocks, table_bboxes

    def _is_significant_visual(self, rect, page_height, page_width):
        if rect.y1 < (page_height * 0.15):
            return False
        if rect.y0 > (page_height * 0.85):
            return False

        if rect.width < 30 or rect.height < 30:
            return False

        if rect.height > 0 and (rect.width / rect.height) > 10:
            return False
        if rect.width > 0 and (rect.height / rect.width) > 10:
            return False

        if rect.width > (page_width * 0.8) and rect.height > (page_height * 0.5):
            return False

        return True

    def _has_complex_visuals(self, fitz_page, table_bboxes):
        page_width = fitz_page.rect.width
        page_height = fitz_page.rect.height

        drawings = fitz_page.get_drawings()
        significant_drawings = 0

        for d in drawings:
            # --- FIX: Safe rect extraction ---
            # d["rect"] is usually a tuple/list, but if it is already a Rect or something else,
            # fitz.Rect() constructor handles it.
            try:
                if "rect" in d:
                    r = fitz.Rect(d["rect"])
                else:
                    # Fallback for older versions or odd data structures
                    r = fitz.Rect(d)
            except Exception:
                continue
            # ----------------------------------

            if any(table.intersects(r) for table in table_bboxes):
                continue
            if self._is_significant_visual(r, page_height, page_width):
                significant_drawings += 1

        images = fitz_page.get_images(full=True)
        significant_images = 0

        for img in images:
            xref = img[0]
            try:
                img_rects = fitz_page.get_image_rects(xref)
            except Exception:
                img_rects = []

            for r in img_rects:
                if any(table.intersects(r) for table in table_bboxes):
                    continue
                if r.width > 50 and r.height > 50:
                    if self._is_significant_visual(r, page_height, page_width):
                        significant_images += 1

        if significant_images > 0:
            return True

        if significant_drawings > 10:
            return True

        return False

    def parse(self):
        structured_pages = []

        # --- OPTIMIZATION: Open files ONCE ---
        try:
            doc = fitz.open(self.pdf_path)
            
            with pdfplumber.open(self.pdf_path) as plumber_pdf:
                
                for page_idx, page in enumerate(doc):
                    combined_elements = []

                    # Safe plumber page access
                    if page_idx < len(plumber_pdf.pages):
                        plumber_page = plumber_pdf.pages[page_idx]
                        table_blocks, table_bboxes = self._process_tables_as_images(page_idx, page, plumber_page)
                    else:
                        table_blocks, table_bboxes = [], []

                    combined_elements.extend(table_blocks)

                    if self._has_complex_visuals(page, table_bboxes):
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                        img_bytes = pix.tobytes("png")
                        full_page_img_id = self._save_image(img_bytes, page_idx + 1)

                        if full_page_img_id:
                            combined_elements.append({
                                "type": "visual_context_placeholder",
                                "content": f"[Visual Context: Diagram detected. See full page image {full_page_img_id}]",
                                "image_id": full_page_img_id,
                                "top": 0,
                            })

                    blocks = page.get_text("dict").get("blocks", [])
                    for block in blocks:
                        if block.get("type") == 0:
                            block_rect = fitz.Rect(block.get("bbox"))

                            if not any(table.intersects(block_rect) for table in table_bboxes):
                                raw_text = " ".join(span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", []))
                                clean_content = self._clean_text(raw_text)

                                if clean_content:
                                    combined_elements.append({
                                        "type": "text",
                                        "content": clean_content,
                                        "top": block.get("bbox")[1],
                                    })

                    combined_elements.sort(key=lambda x: x.get("top", 0))
                    for element in combined_elements:
                        element.pop("top", None)

                    structured_pages.append({
                        "page_number": page_idx + 1,
                        "blocks": combined_elements,
                    })
            
            doc.close()

        except Exception as e:
            print(f"Error during parsing: {e}")
            raise e

        # --- Write markdown summary to output_dir ---
        try:
            from pathlib import Path
            stem = Path(self.pdf_path).stem
            md_path = Path(self.output_dir) / f"{stem}.md"

            md_lines = []
            for p in structured_pages:
                md_lines.append(f"## Page {p['page_number']}")
                for b in p.get("blocks", []):
                    md_lines.append(b.get("content", ""))
                md_lines.append("")

            md_content = "\n\n".join(md_lines)
            md_path.write_text(md_content, encoding="utf-8")
        except Exception as e:
            md_path = None
            print(f"Failed to write markdown: {e}")

        return {
            "document_path": self.pdf_path,
            "pages": structured_pages,
            "images": self.images,
            "markdown_path": str(md_path) if md_path else None,
            "metadata": {
                "num_pages": len(structured_pages),
                "parser": "deterministic",
            },
        }

