from metadata_extraction.PdfDataSegment import PdfDataSegment


class SemanticPredictionData:
    pdf_data_segments: list[PdfDataSegment]

    def get_text(self):
        return " ".join([x.text for x in self.pdf_tags])

    # @staticmethod
    # def from_text(text: str):
    #     return SemanticPredictionData(pdf_data_segments=[PdfTagData.from_text(text)])
    #
    # @staticmethod
    # def from_texts(texts: list[str]):
    #     return [SemanticPredictionData.from_text(text) for text in texts]
