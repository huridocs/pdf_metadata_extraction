from copy import deepcopy

from pdf_features.PdfToken import PdfToken
from pdf_token_type_labels.TokenType import TokenType

from pdf_topic_classification.TextExtractionMethod import TextExtractionMethod


class OnlyTitlesMethod(TextExtractionMethod):
    def get_titles(self, text_length: int) -> list[PdfToken]:
        total_text = ""
        titles_tokens = [token for paragraph in self.pdf_paragraphs
                         for token in paragraph.tokens if token.token_type == TokenType.TITLE]

        first_title_tokens: list[PdfToken] = list()

        for pdf_token in titles_tokens[:7]:
            if len(total_text + " " + pdf_token.content) > text_length:
                break

            total_text += " " + pdf_token.content
            if len(total_text) > text_length:
                break

            first_title_tokens.append(pdf_token)

        return first_title_tokens

    def get_pdf_tokens(self) -> list[PdfToken]:
        return self.get_titles(500)
