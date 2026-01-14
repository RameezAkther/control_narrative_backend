from .deterministic_parser import DeterministicPDFParser

# Docling parser is optional; import lazily and fail gracefully if dependencies missing
try:
    from .docling_parser import DoclingPDFParser
except Exception:
    DoclingPDFParser = None

__all__ = ["DeterministicPDFParser", "DoclingPDFParser"]
