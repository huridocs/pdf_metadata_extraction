import json
import os
from collections import Counter
from os.path import join
from pathlib import Path

import numpy as np
from pdf_token_type_labels.TokenType import TokenType

from data.ExtractionIdentifier import ExtractionIdentifier
from data.PdfDataSegment import PdfDataSegment
import lightgbm as lgb


class FastSegmentSelector:
    def __init__(self, extraction_identifier: ExtractionIdentifier):
        self.extraction_identifier = extraction_identifier
        self.text_types = [TokenType.TEXT, TokenType.LIST, TokenType.TITLE]
        self.previous_words, self.next_words, self.text_segments = [], [], []

        self.fast_segment_selector_path = Path(join(self.extraction_identifier.get_path(), "fast_segment_selector"))
        if not self.fast_segment_selector_path.exists():
            os.makedirs(self.fast_segment_selector_path, exist_ok=True)

        self.previous_words_path = join(self.fast_segment_selector_path, "previous_words.txt")
        self.next_words_path = join(self.fast_segment_selector_path, "next_words.txt")
        self.model_path = join(self.fast_segment_selector_path, "lightgbm_model.txt")

    def get_features(self, segment: PdfDataSegment):
        features = list()
        text = segment.text_content

        index = self.text_segments.index(segment)
        previous_segment_text = self.text_segments[index - 1].text_content if index > 0 else ""
        next_segment_text = self.text_segments[index + 1].text_content if index + 1 < len(self.text_segments) else ""

        for word in self.previous_words:
            features.append(1 if word in previous_segment_text.lower() else 0)

        for word in self.next_words:
            features.append(1 if word in next_segment_text.lower() else 0)

        features.append(len([x for x in text if x == ","]) / len(text) if text else 0)

        return features

    @staticmethod
    def get_most_common_words(train_segments):
        counter = Counter()
        for segment in train_segments:
            counter.update(segment.text_content.lower().split())
        return [x[0] for x in counter.most_common(30)]

    @staticmethod
    def get_predictive_common_words(segments):
        most_common_words = FastSegmentSelector.get_most_common_words(segments)
        counter_previous_segment = Counter()
        counter_next_segment = Counter()

        for previous_segment, segment, next_segment in zip(segments, segments[1:], segments[2:]):
            if segment.ml_label:
                counter_previous_segment.update(
                    [x for x in previous_segment.text_content.strip().lower().split() if x not in most_common_words]
                )
                counter_next_segment.update(
                    [x for x in next_segment.text_content.strip().lower().split() if x not in most_common_words]
                )
                break

        return ([x[0] for x in counter_previous_segment.most_common(3)], [x[0] for x in counter_next_segment.most_common(3)])

    def create_model(self, segments: list[PdfDataSegment]):
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.previous_words, self.next_words = self.get_predictive_common_words(self.text_segments)

        Path(self.previous_words_path).write_text(json.dumps(self.previous_words))
        Path(self.next_words_path).write_text(json.dumps(self.next_words))

        x, y = self.get_x_y(segments)

        train_data = lgb.Dataset(x, y)
        num_round = 50
        light_gbm_model = lgb.train({}, train_data, num_round)
        light_gbm_model.save_model(self.model_path)

    def get_x_y(self, segments):
        x_rows = []
        y = []

        for segment in segments:
            x_rows.append(self.get_features(segment))
            y.append(segment.ml_label)

        x_train = np.zeros((len(x_rows), len(x_rows[0]) if x_rows else 0))
        for i, v in enumerate(x_rows):
            x_train[i] = v

        return x_train, y

    def predict(self, segments):
        self.text_segments = [x for x in segments if x.segment_type in self.text_types]
        self.load_repeated_words()

        x, y = self.get_x_y(segments)

        model = lgb.Booster(model_file=self.model_path)
        predictions = model.predict(x)

        return [segment for i, segment in enumerate(segments) if predictions[i] > 0.5]

    def load_repeated_words(self):
        try:
            self.previous_words = json.loads(Path(self.previous_words_path).read_text())
            self.next_words = json.loads(Path(self.next_words_path).read_text())
        except:
            self.previous_words = []
            self.next_words = []
