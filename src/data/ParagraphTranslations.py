from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from pydantic import BaseModel

from data.ParagraphTranslation import ParagraphTranslation


class ParagraphTranslations(BaseModel):
    position: int
    translations: list[ParagraphTranslation]

    @staticmethod
    def from_paragraphs_from_language(position, paragraphs_from_language: ParagraphsFromLanguage):
        return ParagraphTranslations(
            position=position,
            translations=[
                ParagraphTranslation.from_paragraph(paragraph) for paragraph in paragraphs_from_language.paragraphs
            ],
        )
