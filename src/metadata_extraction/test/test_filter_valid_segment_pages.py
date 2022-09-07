import os
import shutil
from os.path import join
from typing import List
from unittest import TestCase

from data.LabeledData import LabeledData
from data.PredictionData import PredictionData
from data.SegmentBox import SegmentBox
from metadata_extraction.FilterValidSegmentsPages import FilterValidSegmentPages

DOCKER_VOLUME_PATH = (
    f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))}/docker_volume"
)


class TestFilterValidSegmentPages(TestCase):
    @staticmethod
    def get_labeled_data(page_numbers: List[int]):
        return LabeledData(
            tenant="",
            property_name="",
            xml_file_name="",
            language_iso="",
            label_text="",
            page_width=0,
            page_height=0,
            xml_segments_boxes=[
                SegmentBox(left=0, top=0, width=0, height=0, page_number=page_number) for page_number in page_numbers
            ],
            label_segments_boxes=[
                SegmentBox(left=0, top=0, width=0, height=0, page_number=page_number) for page_number in page_numbers
            ],
        )

    @staticmethod
    def get_prediction_data(page_numbers: List[int]):
        return PredictionData(
            tenant="",
            property_name="",
            xml_file_name="",
            page_width=0,
            page_height=0,
            xml_segments_boxes=[
                SegmentBox(left=0, top=0, width=0, height=0, page_number=page_number) for page_number in page_numbers
            ],
        )

    def test_get_valid_pages_one_document(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        labeled_data = self.get_labeled_data(page_numbers=[1, 3, 6])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data])
        self.assertEqual([1, 2, 3, 4, 5, 6], pages_per_document[0])

    def test_get_valid_pages_two_documents(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        labeled_data_1 = self.get_labeled_data(page_numbers=[1, 3, 6])
        labeled_data_2 = self.get_labeled_data(page_numbers=[2, 4])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([1, 2, 3, 4, 5, 6], pages_per_document[0])
        self.assertEqual([1, 2, 3, 4], pages_per_document[1])

    def test_get_valid_pages_from_the_end(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        labeled_data_1 = self.get_labeled_data(page_numbers=[4, 5])
        labeled_data_2 = self.get_labeled_data(page_numbers=[10, 11, 12])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([2, 3, 4, 5], pages_per_document[0])
        self.assertEqual([9, 10, 11, 12], pages_per_document[1])

    def test_get_valid_pages_from_the_end_different_page_numbers(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        labeled_data_1 = self.get_labeled_data(page_numbers=[10, 11, 12, 13])
        labeled_data_2 = self.get_labeled_data(page_numbers=[2])

        pages_per_document = filter_valid_segment_pages.for_training([labeled_data_1, labeled_data_2])
        self.assertEqual([9, 10, 11, 12, 13], pages_per_document[0])
        self.assertEqual([1, 2], pages_per_document[1])

    def test_get_valid_pages_for_prediction(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        labeled_data = self.get_labeled_data(page_numbers=[1, 2, 3])
        prediction_data = self.get_prediction_data(page_numbers=[2])

        filter_valid_segment_pages.for_training([labeled_data])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_for_prediction_from_the_end(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        labeled_data = self.get_labeled_data(page_numbers=[10, 11, 12, 13])
        prediction_data = self.get_prediction_data(page_numbers=[2])

        filter_valid_segment_pages.for_training([labeled_data])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_for_prediction_void(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        filter_valid_segment_pages.for_training([])
        filter_valid_segment_pages.for_prediction([])

        prediction_data = self.get_prediction_data(page_numbers=[2])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])

    def test_get_valid_pages_only_prediction(self):
        tenant = "tenant_filter"
        property_name = "property_filter"

        shutil.rmtree(join(DOCKER_VOLUME_PATH, tenant), ignore_errors=True)

        filter_valid_segment_pages = FilterValidSegmentPages(tenant, property_name)
        prediction_data = self.get_prediction_data(page_numbers=[2])
        pages_per_document = filter_valid_segment_pages.for_prediction([prediction_data])
        self.assertEqual(1, len(pages_per_document))
        self.assertEqual([1, 2], pages_per_document[0])
