import os
import shutil
from unittest import TestCase

from data.SegmentBox import SegmentBox
from data.SegmentationData import SegmentationData
from information_extraction.XmlFile import XmlFile

DOCKER_VOLUME_PATH = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))}/docker_volume"


class TestXmlFile(TestCase):
    test_file_path = (
        f"{DOCKER_VOLUME_PATH}/tenant_test/property_name/xml_to_train/test.xml"
    )

    def test_save_xml_to_train(self):
        tenant = "tenant_save"
        property_name = "property_save"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                tenant=tenant,
                property_name=property_name,
                to_train=True,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        self.assertTrue(os.path.exists(f"{base_path}/xml_to_train/test.xml"))

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    def test_save_xml_to_predict(self):
        tenant = "tenant_save"
        property_name = "property_save"

        base_path = f"{DOCKER_VOLUME_PATH}/{tenant}/{property_name}"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                tenant=tenant,
                property_name=property_name,
                to_train=False,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        self.assertTrue(os.path.exists(f"{base_path}/xml_to_predict/test.xml"))

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    def test_get_segments(self):
        tenant = "tenant_save"
        property_name = "property_save"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

        segmentation_data = SegmentationData(
            page_width=612,
            page_height=792,
            xml_segments_boxes=[
                SegmentBox(
                    left=495.1,
                    top=42.6323,
                    width=56.96199999999999,
                    height=18.2164,
                    page_number=1,
                ),
                SegmentBox(
                    left=123.38,
                    top=48.1103,
                    width=82.9812,
                    height=12.7624,
                    page_number=1,
                ),
                SegmentBox(
                    left=123.38,
                    top=72.8529,
                    width=148.656,
                    height=17.895700000000005,
                    page_number=1,
                ),
                SegmentBox(
                    left=123.38,
                    top=245.184,
                    width=317.406,
                    height=27.5377,
                    page_number=1,
                ),
            ],
            label_segments_boxes=[
                SegmentBox(left=125, top=247, width=319, height=29, page_number=1)
            ],
        )

        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                tenant=tenant,
                property_name=property_name,
                to_train=True,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        segments = xml_file.get_segments(segmentation_data)

        self.assertEqual(612, xml_file.segments[0].page_width)
        self.assertEqual(792, segments[0].page_height)
        self.assertEqual(
            1, len([segment for segment in segments if segment.ml_class_label == 1])
        )
        self.assertEqual("United Nations", segments[0].text_content)
        self.assertEqual("A/INF/76/1", segments[1].text_content)
        self.assertEqual("General Assembly", segments[2].text_content)
        self.assertEqual(
            "Opening dates of forthcoming regular sessions of the General Assembly and of the general debate",
            [segment for segment in segments if segment.ml_class_label == 1][
                0
            ].text_content,
        )

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    def test_get_segments_different_page_size_scale(self):
        tenant = "tenant_save"
        property_name = "property_save"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        segmentation_data = SegmentationData(
            page_width=1,  # 612
            page_height=2,  # 396
            xml_segments_boxes=[
                SegmentBox(
                    left=0, top=0.3282828, width=1, height=0.1767676, page_number=2
                ),
            ],
            label_segments_boxes=[
                SegmentBox(
                    left=0.49019,
                    top=0.37878,
                    width=0.008169,
                    height=0.0126,
                    page_number=2,
                )
            ],
        )
        with open(self.test_file_path, "rb") as file:
            xml_file = XmlFile(
                tenant=tenant,
                property_name=property_name,
                to_train=False,
                xml_file_name="test.xml",
            )

            xml_file.save(file=file.read())

        segments = xml_file.get_segments(segmentation_data)

        labeled_segments = [
            segment for segment in segments if segment.ml_class_label == 1
        ]

        self.assertEqual(612, segments[0].page_width)
        self.assertEqual(792, segments[0].page_height)
        self.assertEqual(1, len(labeled_segments))
        self.assertEqual(
            "a In accordance with paragraph", labeled_segments[0].text_content[:30]
        )
        self.assertEqual("every four years.", labeled_segments[0].text_content[-17:])

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    def test_get_segments_when_no_file(self):
        tenant = "tenant_save"
        property_name = "property_save"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
        segmentation_data = SegmentationData(
            page_width=1,  # 612
            page_height=2,  # 396
            xml_segments_boxes=[],
            label_segments_boxes=[],
        )

        xml_file = XmlFile(
            tenant=tenant,
            property_name=property_name,
            to_train=True,
            xml_file_name="test.xml",
        )

        segments = xml_file.get_segments(segmentation_data)

        self.assertEqual(0, len(segments))

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

    def test_get_segments_should_be_empty_when_no_file_because_different_property_name(
        self,
    ):
        tenant = "tenant_save"
        property_name = "property_save"

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)

        segmentation_data = SegmentationData(
            page_width=1,  # 612
            page_height=2,  # 396
            xml_segments_boxes=[
                SegmentBox(
                    left=0, top=0.3282828, width=1, height=0.1767676, page_number=2
                ),
            ],
            label_segments_boxes=[
                SegmentBox(
                    left=0.49019,
                    top=0.37878,
                    width=0.008169,
                    height=0.0126,
                    page_number=2,
                )
            ],
        )
        with open(self.test_file_path, "rb") as file:
            XmlFile(
                tenant=tenant,
                property_name="different_property_name",
                to_train=False,
                xml_file_name="test.xml",
            ).save(file=file.read())

        xml_file = XmlFile(
            tenant=tenant,
            property_name=property_name,
            to_train=False,
            xml_file_name="test.xml",
        )

        segments = xml_file.get_segments(segmentation_data)

        self.assertEqual(0, len(segments))

        shutil.rmtree(f"{DOCKER_VOLUME_PATH}/{tenant}", ignore_errors=True)
