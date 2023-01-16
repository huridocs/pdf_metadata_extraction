import json
import os
from os.path import join, exists
from typing import List

from config import DATA_PATH
from data.LabeledData import LabeledData
from data.PredictionData import PredictionData


class FilterValidSegmentPages:
    def __init__(self, tenant: str, property_name: str):
        self.labeled_data_json_path = join(DATA_PATH, tenant, property_name, "filter_pages.json")
        self.start_gaps = []
        self.end_gaps = []
        self.valid_pages_ranges = []

    def get_valid_pages(self, total_number_pages_per_document):
        if min(self.start_gaps) <= min(self.end_gaps):
            start = min(self.start_gaps)
            end = max(self.start_gaps) + max(self.valid_pages_ranges)
            return [self.get_range(start, end, number_of_pages) for number_of_pages in total_number_pages_per_document]

        valid_page_numbers_from_the_end = []

        for number_of_pages in total_number_pages_per_document:
            start = number_of_pages - max(self.end_gaps) - max(self.valid_pages_ranges)
            end = number_of_pages - min(self.end_gaps)
            valid_page_numbers_from_the_end.append(self.get_range(start, end, number_of_pages))

        return valid_page_numbers_from_the_end

    def for_training(self, labeled_data_list: List[LabeledData]):
        if not exists(os.path.dirname(self.labeled_data_json_path)):
            os.makedirs(os.path.dirname(self.labeled_data_json_path))

        with open(self.labeled_data_json_path, "w") as file:
            json.dump([x.dict() for x in labeled_data_list], file)

            self.set_parameters(labeled_data_list)
            pages_list = [[x.page_number for x in labeled_data.xml_segments_boxes] for labeled_data in labeled_data_list]
            total_number_pages_per_document = [max(pages) if pages else 1000 for pages in pages_list]
            return self.get_valid_pages(total_number_pages_per_document)

    def for_prediction(self, prediction_data_list: List[PredictionData]):
        try:
            with open(self.labeled_data_json_path, "r") as file:
                labeled_data_dict = json.load(file)
                labeled_data_list = [LabeledData(**x) for x in labeled_data_dict]
                self.set_parameters(labeled_data_list)
        except:
            self.set_parameters([])

        pages_list = [
            [x.page_number for x in prediction_data.xml_segments_boxes] for prediction_data in prediction_data_list
        ]
        number_pages_per_document = [max(pages) if pages else 1000 for pages in pages_list]

        return self.get_valid_pages(number_pages_per_document)

    def set_parameters(self, labeled_data_list):
        if not labeled_data_list or not labeled_data_list[0].label_segments_boxes:
            self.start_gaps = [0]
            self.end_gaps = [0]
            self.valid_pages_ranges = [0]
            return

        for labeled_data in labeled_data_list:
            if not labeled_data.label_segments_boxes:
                continue

            pages = [x.page_number for x in labeled_data.xml_segments_boxes]
            number_of_pages = max(pages) if pages else 1000
            valid_page_numbers = [x.page_number for x in labeled_data.label_segments_boxes]
            self.valid_pages_ranges.append(max(valid_page_numbers) - min(valid_page_numbers) + 1)
            self.start_gaps.append(min(valid_page_numbers) - 1)
            self.end_gaps.append(number_of_pages - max(valid_page_numbers))

        if not self.valid_pages_ranges:
            self.start_gaps = [0]
            self.end_gaps = [0]
            self.valid_pages_ranges = [0]

    @staticmethod
    def get_range(start: int, end: int, number_pages: int):
        start = max(0, start - 1)
        end = min(end + 1, number_pages)
        return list(range(start + 1, end + 1))
