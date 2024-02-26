from copy import deepcopy
from os.path import join
from pathlib import Path

from pdf_features.PdfFont import PdfFont
from pdf_features.PdfToken import PdfToken
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

from config import ROOT_PATH
from pdf_topic_classification.TextExtractionMethod import TextExtractionMethod


class SummariesMethod(TextExtractionMethod):
    def get_pdf_tokens(self) -> list[PdfToken]:
        padding_token = self.get_padding_token()
        if not self.pdf_paragraphs:
            print(":::::::::::::::::::::: no summary")
            return [padding_token]

        summary = Path(join(ROOT_PATH, "data", "summaries", f"{self.pdf_paragraphs[0].pdf_name}.txt")).read_text()

        if summary:
            padding_token.content = summary

        return [padding_token]

    @staticmethod
    def get_padding_token() -> PdfToken:
        return PdfToken(
            1,
            "pad_token",
            "no text",
            PdfFont("pad_font_id", False, False, 0.0, "#000000"),
            1,
            Rectangle(0, 0, 0, 0),
            TokenType.TEXT,
        )
