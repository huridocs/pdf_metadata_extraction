import pymongo

from data.ExtractionId import ExtractionId
from data.Label import Label
from data.SegmentedXml import SegmentedXml


class LabeledData:
    def __init__(self,
                 label_id: str,
                 extraction_id: ExtractionId,
                 segmented_xml: SegmentedXml,
                 label: Label):
        self.label_id = label_id
        self.extraction_id = extraction_id
        self.segmented_xml = segmented_xml
        self.label = label
        self.client = pymongo.MongoClient('mongodb://mongo:27017')
        self.db = self.client['pdf_information_extraction']

    def save(self):
        document = {'label_id': self.label_id,
                    'extraction_id': self.extraction_id.extraction_id,
                    'xml_segments_boxes': self.segmented_xml.segments_boxes.to_json(),
                    'label_text': self.label.text,
                    'label_segments_boxes': self.label.segments_boxes.to_json()}

        if self.extraction_id.tenant:
            document['tenant'] = self.extraction_id.tenant

        self.db.labeleddata.insert_one(document)