from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from pydantic import BaseModel


class ParagraphTranslation(BaseModel):
    language: str
    text: str
    needs_user_review: bool

    @staticmethod
    def from_paragraph_features(language: str, paragraph_features: ParagraphFeatures):
        return ParagraphTranslation(
            language=language,
            text=paragraph_features.original_text,
            needs_user_review=False if paragraph_features.original_text else True,
        )


class ParagraphTranslations(BaseModel):
    position: int
    translations: list[ParagraphTranslation]

    def append_language(self, language: str, paragraph_features: ParagraphFeatures):
        self.translations.append(ParagraphTranslation.from_paragraph_features(language, paragraph_features))

    @staticmethod
    def from_features(index: int, language: str, paragraph_features: ParagraphFeatures):
        return ParagraphTranslations(
            position=index + 1,
            translations=[ParagraphTranslation.from_paragraph_features(language, paragraph_features)],
        )


class ParagraphsTranslations(BaseModel):
    key: str
    main_language: str = ""
    available_languages: list[str] = list()
    paragraphs: list[ParagraphTranslations] = list()

    @staticmethod
    def from_paragraphs_from_languages(
        key: str, paragraphs_from_languages: list[ParagraphsFromLanguage]
    ) -> "ParagraphsTranslations":
        languages = [x.language for x in paragraphs_from_languages]
        main_paragraphs = [x for x in paragraphs_from_languages if x.is_main_language][0]
        main_language = main_paragraphs.language
        other_languages = [x for x in paragraphs_from_languages if not x.is_main_language]

        paragraphs: list[ParagraphTranslations] = [
            ParagraphTranslations.from_features(i, main_language, x) for i, x in enumerate(main_paragraphs.paragraphs)
        ]

        for other_language in other_languages:
            for paragraph_translation, paragraph in zip(paragraphs, other_language.paragraphs):
                paragraph_translation.append_language(other_language.language, paragraph)

        paragraphs_translations = ParagraphsTranslations(
            key=key, main_language=main_paragraphs.language, available_languages=languages, paragraphs=paragraphs
        )
        return paragraphs_translations
