from data.PdfDataSegment import PdfDataSegment
from extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import FilterSegmentsMethod


class NoFilter(FilterSegmentsMethod):

    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        return pdf_data_segments
