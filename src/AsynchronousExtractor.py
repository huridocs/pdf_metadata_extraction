from data.ExtractionStatus import ExtractionStatus


class AsynchronousExtractor:
    def __init__(self, extraction_identifier):
        self.extraction_identifier = extraction_identifier

    def get_status(self):
        return ExtractionStatus.NO_MODEL
