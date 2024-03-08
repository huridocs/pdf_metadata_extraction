from collections import Counter

from paragraph_extraction_trainer.PdfSegment import PdfSegment
from rapidfuzz import fuzz

from multi_option_extraction.PdfLabels import PdfLabels
from multi_option_extraction.PdfMultiOptionExtractionMethod import PdfMultiOptionExtractionMethod


class FuzzyFirstCleanTo80Label(PdfMultiOptionExtractionMethod):
    def get_first_appearance(self, pdf_segments: list[PdfSegment], options: list[str]) -> list[str]:
        for pdf_segment in pdf_segments:
            for ratio_threshold in range(100, 79, -10):
                for option in options:
                    if fuzz.partial_ratio(option, pdf_segment.text_content.lower()) >= ratio_threshold:
                        return [option]

        return []

    def predict(self, pdfs_labels: list[PdfLabels]):
        predictions = list()
        clean_options = self.get_cleaned_options(self.options)
        for pdf_label in pdfs_labels:
            pdf_segments = [PdfSegment.from_pdf_tokens(x.tokens) for x in pdf_label.paragraphs]
            prediction = self.get_first_appearance(pdf_segments, clean_options)
            if prediction:
                predictions.append([self.options[clean_options.index(prediction[0])]])
            else:
                predictions.append([])

        return predictions

    def train(self, pdfs_labels: list[PdfLabels]):
        pass

    def get_cleaned_options(self, options: list[str]):
        options = [x.lower() for x in options]
        words_counter = Counter()
        for option in options:
            words_counter.update(option.split())

        clean_options = list()
        for option in options:
            clean_options.append(option)
            for word, count in words_counter.most_common():
                if count == 1:
                    continue

                if word not in option:
                    continue

                if clean_options[-1].replace(word, "").strip() != "":
                    clean_options[-1] = clean_options[-1].replace(word, "").strip()

        return clean_options
