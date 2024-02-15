from copy import deepcopy

from paragraph_extraction_trainer.Paragraph import Paragraph
from pdf_features.PdfToken import PdfToken

from data.SemanticPredictionData import SemanticPredictionData
from pdf_topic_classification.TextExtractionMethod import TextExtractionMethod


class TextAtTheBeginningMethod(TextExtractionMethod):
    def get_first_tokens(self, text_length: int) -> list[PdfToken]:
        total_text = ""
        first_tokens: list[PdfToken] = list()
        for pdf_paragraph in self.pdf_paragraphs:
            for pdf_token in pdf_paragraph.tokens:
                if len(total_text + " " + pdf_token.content) > text_length:
                    break

                total_text += " " + pdf_token.content
                first_tokens.append(pdf_token)

        if len(first_tokens) < 2 and self.pdf_paragraphs and self.pdf_paragraphs[0].tokens:
            fist_token = deepcopy(self.pdf_paragraphs[0].tokens[0])
            fist_token.content = fist_token.content[:text_length]
            return [fist_token]

        return first_tokens

    def get_text(self) -> list[SemanticPredictionData]:
        pdf_tokens: list[PdfToken] = self.get_first_tokens(750)
        return [SemanticPredictionData.from_text(x.content) for x in pdf_tokens]
