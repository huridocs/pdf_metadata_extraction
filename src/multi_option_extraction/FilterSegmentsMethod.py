from abc import ABC, abstractmethod

from metadata_extraction.PdfData import PdfData
from metadata_extraction.PdfDataSegment import PdfDataSegment
from multi_option_extraction.data.MultiOptionData import MultiOptionData
from multi_option_extraction.data.MultiOptionSample import MultiOptionSample


class FilterSegmentsMethod(ABC):
    @abstractmethod
    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        pass

    def filter(self, multi_option_data):
        filtered_samples: list[MultiOptionSample] = list()
        for sample in multi_option_data.samples:
            filtered_pdf_data = PdfData(
                pdf_features=sample.pdf_data.pdf_features,
                file_name=sample.pdf_data.file_name,
                file_type=sample.pdf_data.file_type,
            )

            filtered_pdf_data.pdf_data_segments = self.filter_segments(sample.pdf_data.pdf_data_segments)

            filtered_samples.append(
                MultiOptionSample(pdf_data=filtered_pdf_data, values=sample.values, language_iso=sample.language_iso)
            )

        return MultiOptionData(
            samples=filtered_samples, options=multi_option_data.options, multi_value=multi_option_data.multi_value
        )
