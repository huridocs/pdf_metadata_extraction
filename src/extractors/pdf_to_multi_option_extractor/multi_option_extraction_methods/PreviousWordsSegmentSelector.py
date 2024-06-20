import json
from pathlib import Path
from data.PdfDataSegment import PdfDataSegment

from extractors.pdf_to_multi_option_extractor.multi_option_extraction_methods.FastSegmentSelector import FastSegmentSelector


class PreviousWordsSegmentSelector(FastSegmentSelector):
    def create_model(self, segments: list[PdfDataSegment]):
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.get_predictive_common_words(self.text_segments)

        Path(self.previous_words_path).write_text(json.dumps(self.previous_words))

    def predict(self, segments):
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.load_repeated_words()

        predicted_segments = []
        for segment in self.text_segments:

            index = self.text_segments.index(segment)
            previous_segment_texts = self.clean_texts(self.text_segments[index - 1]) if index > 0 else []

            for word in self.previous_words:
                if word in previous_segment_texts:
                    predicted_segments.append(segment)
                    break

        return predicted_segments
