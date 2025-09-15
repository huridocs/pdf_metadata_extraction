from multilingual_paragraph_extractor.domain.ParagraphFeatures import ParagraphFeatures
from multilingual_paragraph_extractor.domain.ParagraphsFromLanguage import ParagraphsFromLanguage
from multilingual_paragraph_extractor.use_cases.MultilingualParagraphAlignerUseCase import (
    MultilingualParagraphAlignerUseCase,
)
from trainable_entity_extractor.domain.ExtractionIdentifier import ExtractionIdentifier
from trainable_entity_extractor.domain.PdfData import PdfData
from trainable_entity_extractor.domain.SegmentationData import SegmentationData
from trainable_entity_extractor.use_cases.XmlFile import XmlFile

from config import MODELS_DATA_PATH, PARAGRAPH_EXTRACTION_NAME
from domain.ParagraphExtractorTask import ParagraphExtractorTask
from ports.PersistenceRepository import PersistenceRepository


class ParagraphExtractorUseCase:
    def __init__(self, extraction_identifier: ExtractionIdentifier, persistence_repository: PersistenceRepository):
        self.extraction_identifier = extraction_identifier
        self.persistence_repository = persistence_repository

    def save_paragraphs_from_languages(self) -> tuple[bool, str]:
        paragraph_extraction_data = self.persistence_repository.load_paragraph_extraction_data(self.extraction_identifier)
        if not paragraph_extraction_data:
            return False, "No data to extract paragraphs"

        paragraphs_from_languages = self.get_paragraphs_from_languages(paragraph_extraction_data)

        aligner_use_case = MultilingualParagraphAlignerUseCase(self.extraction_identifier)
        aligner_use_case.align_languages(paragraphs_from_languages)

        for paragraphs_from_language in paragraphs_from_languages:
            self.persistence_repository.save_paragraphs_from_language(
                self.extraction_identifier, paragraphs_from_language.to_db()
            )

        return True, ""

    def get_paragraphs_from_languages(self, paragraph_extraction_data):
        paragraphs_from_languages: list[ParagraphsFromLanguage] = list()
        for xml_segments in paragraph_extraction_data.xmls:
            segmentation_data = SegmentationData(
                page_width=0, page_height=0, xml_segments_boxes=xml_segments.xml_segments_boxes, label_segments_boxes=[]
            )
            xml_file = XmlFile(
                extraction_identifier=self.extraction_identifier, to_train=True, xml_file_name=xml_segments.xml_file_name
            )
            pdf_data = PdfData.from_xml_file(xml_file, segmentation_data)
            paragraphs_from_language = ParagraphsFromLanguage(
                language=xml_segments.language,
                paragraphs=[ParagraphFeatures.from_pdf_data(pdf_data, x) for x in pdf_data.pdf_data_segments],
                is_main_language=xml_segments.is_main_language,
            )
            paragraphs_from_languages.append(paragraphs_from_language)
            xml_file.delete()

        return paragraphs_from_languages

    @staticmethod
    def execute_task(task: ParagraphExtractorTask, persistence_repository: PersistenceRepository) -> tuple[bool, str]:
        extraction_identifier = ExtractionIdentifier(
            run_name=PARAGRAPH_EXTRACTION_NAME, extraction_name=task.key, output_path=MODELS_DATA_PATH
        )
        extractor = ParagraphExtractorUseCase(extraction_identifier, persistence_repository)
        return extractor.save_paragraphs_from_languages()
