from pathlib import Path
from PIL import Image
import imagehash
import tempfile
import logging

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem

logging.basicConfig(level=logging.INFO)

class DoclingPDFParser:
    """
    Accurate parser using Docling. It saves unique images and exports markdown.
    Returns markdown, sections (split from markdown), images map and metadata.
    """

    def __init__(self, pdf_path, output_dir=None, phash_threshold=5):
        self.pdf_path = pdf_path
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="parsed_docling_")
        self.images_dir = Path(self.output_dir) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.phash_threshold = phash_threshold

    def _phash(self, img: Image.Image):
        return imagehash.phash(img)

    def parse(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        conv_res = converter.convert(self.pdf_path)

        # Image deduplication using perceptual hash
        DUP_THRESHOLD = self.phash_threshold
        hash_to_filename = {}
        element_to_filename = {}
        picture_counter = 0
        table_counter = 0

        for element, _level in conv_res.document.iterate_items():
            if not isinstance(element, (PictureItem, TableItem)):
                continue

            pil_img = element.get_image(conv_res.document)
            if pil_img is None:
                continue

            img_hash = self._phash(pil_img)

            canonical_name = None
            for saved_hash, saved_name in hash_to_filename.items():
                if img_hash - saved_hash <= DUP_THRESHOLD:
                    canonical_name = saved_name
                    break

            # New unique image
            if canonical_name is None:
                if isinstance(element, TableItem):
                    table_counter += 1
                    canonical_name = f"table-{table_counter:02d}.png"
                else:
                    picture_counter += 1
                    canonical_name = f"picture-{picture_counter:02d}.png"

                img_path = self.images_dir / canonical_name
                pil_img.save(img_path, "PNG")

                hash_to_filename[img_hash] = canonical_name

            element_to_filename[id(element)] = canonical_name

        # Export markdown (plain text)
        markdown_content = conv_res.document.export_to_markdown()

        # Append image references
        image_section = "\n\n## 📸 Images\n\n"
        for filename in sorted(hash_to_filename.values()):
            rel_path = f"images/{filename}"
            image_section += f"![{filename}]({rel_path})\n\n"

        markdown_content += image_section

        # Save markdown file to output_dir
        try:
            stem = Path(self.pdf_path).stem
            md_path = Path(self.output_dir) / f"{stem}.md"
            md_path.write_text(markdown_content, encoding="utf-8")
        except Exception as e:
            md_path = None
            print(f"Failed to write docling markdown: {e}")

        # Split markdown into sections by top-level headings
        sections = []
        current_title = "Document"
        current_lines = []

        for line in markdown_content.splitlines():
            if line.startswith("#"):
                if current_lines:
                    sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})
                # Use heading text as new title
                current_title = line.lstrip('#').strip() or "Section"
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})

        # Build file paths map
        images_map = {v: str((self.images_dir / v).resolve()) for v in hash_to_filename.values()}

        return {
            "document_path": self.pdf_path,
            "markdown": markdown_content,
            "markdown_path": str(md_path) if md_path else None,
            "sections": sections,
            "images": images_map,
            "metadata": {
                "num_sections": len(sections),
                "num_images": len(hash_to_filename),
                "parser": "docling",
            },
        }
