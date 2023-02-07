from pydantic import BaseModel

from metadata_extraction.PdfFeatures.PdfTag import PdfTag


class SemanticExtractionData(BaseModel):
    text: str
    pdf_tags: list[PdfTag]
    language_iso: str
