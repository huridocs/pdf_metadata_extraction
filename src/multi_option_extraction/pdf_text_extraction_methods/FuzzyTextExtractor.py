from paragraph_extraction_trainer.PdfSegment import PdfSegment
from pdf_features.PdfToken import PdfToken
from rapidfuzz import fuzz

from multi_option_extraction.PdfTextExtractionMethod import PdfTextExtractionMethod

ratio_threshold = 75


class FuzzyTextExtractor(PdfTextExtractionMethod):
    def get_pdf_tokens(self) -> list[PdfToken]:
        pdf_tokens = []
        for paragraph in self.pdf_paragraphs:
            pdf_segment = PdfSegment.from_pdf_tokens(paragraph.tokens)
            for option in self.options:
                ratio = fuzz.partial_ratio(option.lower(), pdf_segment.text_content.lower())
                if ratio >= ratio_threshold:
                    pdf_tokens.extend(paragraph.tokens)
                    break

        return pdf_tokens
