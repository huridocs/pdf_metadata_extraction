from copy import deepcopy

from pdf_features.PdfToken import PdfToken
from multi_option_extraction.PdfTextExtractionMethod import PdfTextExtractionMethod


class CleanBeginningDot2500(PdfTextExtractionMethod):
    def get_first_tokens(self, text_length: int) -> list[PdfToken]:
        total_text = ""
        first_tokens: list[PdfToken] = list()
        for pdf_paragraph in self.pdf_paragraphs:
            for pdf_token in pdf_paragraph.tokens:
                if len(total_text + " " + pdf_token.content) > text_length:
                    break
                token_copy = self.clean_content_pdf_token(pdf_token)
                if "." == pdf_token.content[-1]:
                    token_copy.content += "."
                total_text += " " + token_copy.content
                first_tokens.append(token_copy)

        if len(first_tokens) < 2 and self.pdf_paragraphs and self.pdf_paragraphs[0].tokens:
            fist_token = self.clean_content_pdf_token(self.pdf_paragraphs[0].tokens[0])
            fist_token.content = fist_token.content[:text_length]
            return [fist_token]

        return first_tokens

    @staticmethod
    def clean_content_pdf_token(pdf_token):
        token_copy = deepcopy(pdf_token)
        words = list()
        for word in token_copy.content.split():
            clean_word = "".join([x for x in word if x.isalpha()])
            if clean_word:
                words.append(clean_word)
        token_copy.content = " ".join(words)
        return token_copy

    def get_pdf_tokens(self) -> list[PdfToken]:
        return self.get_first_tokens(2500)
