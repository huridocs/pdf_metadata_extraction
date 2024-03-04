from copy import deepcopy

import spacy
from pdf_features.PdfToken import PdfToken
from pdf_topic_classification.TextExtractionMethod import TextExtractionMethod

nlp = spacy.load("en_core_web_md")


class NounsBeginningMethod500V3(TextExtractionMethod):
    def get_first_tokens(self, text_length: int) -> list[PdfToken]:
        total_text = ""
        first_tokens: list[PdfToken] = list()
        for pdf_paragraph in self.pdf_paragraphs:
            for pdf_token in pdf_paragraph.tokens:
                if len(total_text + " " + pdf_token.content) > text_length:
                    break
                token_copy = self.clean_content_pdf_token(pdf_token)

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
        doc = nlp(token_copy.content)
        words = list()
        for x in doc:
            if x.pos_ in ["NOUN", "PRON", "PROPN"]:
                words.append(str(x))
        token_copy.content = " ".join(words)

        return token_copy

    def get_pdf_tokens(self) -> list[PdfToken]:
        return self.get_first_tokens(500)
