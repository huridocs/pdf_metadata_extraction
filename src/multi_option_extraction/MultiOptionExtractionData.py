from dataclasses import dataclass

from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticExtractionData import SemanticExtractionData


@dataclass
class MultiOptionExtractionSample:
    pdf_tags: list[PdfTagData]
    options: list[Option]
    language_iso: str = ""


@dataclass
class MultiOptionExtractionData:
    multi_value: bool
    options: list[Option]
    samples: list[MultiOptionExtractionSample]

    def to_semantic_extraction_data(self) -> list[SemanticExtractionData]:
        semantic_extraction_data_list: list[SemanticExtractionData] = list()

        for sample in self.samples:
            text = " ; ".join([option.label for option in sample.options])
            semantic_extraction_data = SemanticExtractionData(
                text=text, pdf_tags=sample.pdf_tags, language_iso=sample.language_iso
            )
            semantic_extraction_data_list.append(semantic_extraction_data)

        return semantic_extraction_data_list
