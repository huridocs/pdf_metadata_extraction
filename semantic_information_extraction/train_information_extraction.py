import os
from random import choice
from typing import List

from segment_predictor.InformationExtraction import SegmentPredictor
from segment_predictor.Model import Model
import pandas as pd

from segment_predictor.SegmentsBoxes import SegmentsBoxes


def train_information_extraction(name, pdfs, segments_boxes: SegmentsBoxes,  texts: List[str]):
    information_extractions: List[SegmentPredictor] = []

    xml_labeled_documents = []
    for information_extraction in information_extractions:
        xml_name = ''.join(choice('asdfghjklqwertyuiopzxcvbnm') for _ in range(12))
        xml_name += '.xml'
        information_extraction.save_labeled_document(xml_name)
        xml_labeled_documents.append(xml_name)

    ix_model = Model(f'files/{name}')
    ix_model.save_model(xml_labeled_documents)
    performance = ix_model.get_performance(xml_labeled_documents)

    model = ix_model.load_model()

    labeled_data = list()
    for index, information_extraction in enumerate(information_extractions):
        information_extraction.set_predictions(model)
        input_text = ' '.join([x.text for x in information_extraction.get_predicted_segments_tag()])
        labeled_data.append([input_text, texts[index]])

    semantic_labeled_data_frame = pd.DataFrame(labeled_data)
    semantic_labeled_data_frame.columns = ["input_text", "target_text"]

