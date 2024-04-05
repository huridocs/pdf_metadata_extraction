from abc import ABC, abstractmethod

from data.PdfData import PdfData
from data.PdfDataSegment import PdfDataSegment
from data.ExtractionData import ExtractionData
from data.TrainingSample import TrainingSample


class FilterSegmentsMethod(ABC):
    @abstractmethod
    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        pass

    def filter(self, multi_option_data: ExtractionData) -> ExtractionData:
        filtered_samples: list[TrainingSample] = list()
        for sample in multi_option_data.samples:
            filtered_pdf_data = PdfData(
                pdf_features=sample.pdf_data.pdf_features,
                file_name=sample.pdf_data.file_name,
                file_type=sample.pdf_data.file_type,
            )

            filtered_pdf_data.pdf_data_segments = self.filter_segments(sample.pdf_data.pdf_data_segments)

            filtered_samples.append(TrainingSample(pdf_data=filtered_pdf_data, labeled_data=sample.labeled_data))

        return ExtractionData(
            samples=filtered_samples, options=multi_option_data.options, multi_value=multi_option_data.multi_value
        )
