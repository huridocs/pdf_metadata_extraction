from abc import ABC, abstractmethod

from paragraph_extraction_trainer.Paragraph import Paragraph
from pdf_features.PdfToken import PdfToken

from data.Option import Option
from data.PdfTagData import PdfTagData
from data.SemanticPredictionData import SemanticPredictionData
from multi_option_extraction.MultiOptionExtractionData import MultiOptionExtractionSample


class TextExtractionMethod(ABC):
    def __init__(self, pdf_paragraphs: list[Paragraph], options: list[str], multi_option: bool):
        self.pdf_paragraphs: list[Paragraph] = pdf_paragraphs
        self.options = options
        self.multi_option = multi_option

    @abstractmethod
    def get_pdf_tokens(self) -> list[PdfToken]:
        pass

    def get_semantic_prediction_data(self) -> SemanticPredictionData:
        pdf_tokens: list[PdfToken] = self.get_pdf_tokens()
        texts = [x.content for x in pdf_tokens]
        texts = [text.replace("__", "").replace("...", "..") for text in texts]
        texts = [text.replace('"', "'").replace("...", "..") for text in texts]

        if not pdf_tokens or not "".join(texts):
            return SemanticPredictionData(pdf_tags=PdfTagData.from_texts(["no text"]))

        return SemanticPredictionData(pdf_tags=PdfTagData.from_texts(texts))

    def get_labeled_data_sample(self, labels: list[str]) -> MultiOptionExtractionSample:
        values = [Option(id=option, label=option) for option in self.options if option in labels]
        semantic_prediction_data = self.get_semantic_prediction_data()
        return MultiOptionExtractionSample(pdf_tags=semantic_prediction_data.pdf_tags, values=values)
