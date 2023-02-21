import numpy as np
from typing import List, Dict


from metadata_extraction.PdfFeatures.PdfFont import PdfFont
from metadata_extraction.PdfFeatures.PdfTag import PdfTag
from metadata_extraction.PdfFeatures.Rectangle import Rectangle
from metadata_extraction.PdfFeatures.TagType import TagType, TAG_TYPE_BY_INDEX
from segment_selector.LightGBM_30Features_OneHotOneLetter.get_features import PdfAltoXml

tag_type_dict: Dict = {
    "text": 6,
    "title": 5,
    "figure": 4,
    "table": 3,
    "list": 2,
    "footnote": 1,
    "formula": 0,
    "code": 3,
}

tag_type_by_index: Dict = {6: "text", 5: "title", 4: "figure", 3: "table", 2: "list", 1: "footnote", 0: "formula"}


class LightGBM_30Features_OneHotOneLetter:
    def __init__(self, X_train, y_train, model_configs: Dict, model=None, benchmarking=False):
        self.X_train = X_train
        self.y_train = y_train
        self.model_configs = model_configs
        self.model = model
        self.benchmarking = benchmarking
        self.tag_type_counts = {}
        self.wrong_prediction_counts = {}

    def get_predicted_tag_types(
        self, pdfalto_xml, page_tags: List[PdfTag], predicted_tag_types: List[TagType] = list
    ) -> List[TagType]:
        x_rows = list()
        context_size: int = self.model_configs["context_size"]

        for i in range(context_size):
            page_tags.insert(
                0,
                PdfTag(
                    page_tags[0].page_number,
                    "pad_tag",
                    "pad_content",
                    PdfFont("pad_font_id", False, False, 0.0, "b"),
                    -i - 1,
                    -i - 1,
                    Rectangle(0, 0, 0, 0),
                    "pad_type",
                ),
            )

        for i in range(context_size + 1):
            page_tags.append(
                PdfTag(
                    page_tags[0].page_number,
                    "pad_tag",
                    "pad_content",
                    PdfFont("pad_font_id", False, False, 0.0, "b"),
                    -i - 1000,
                    -i - 1000,
                    Rectangle(0, 0, 0, 0),
                    "pad_type",
                )
            )

        for tag_index, tag in enumerate(page_tags):
            if tag_index + (2 * context_size + 2) > len(page_tags):
                continue

            new_data_row = []

            for i in range(2 * context_size + 1):
                new_data_row.extend(
                    pdfalto_xml.get_features_for_given_tags(
                        page_tags[tag_index + i], page_tags[tag_index + i + 1], page_tags
                    )
                )

            x_rows.append(new_data_row)

        X = np.zeros((len(x_rows), len(x_rows[0]) if x_rows else 0))
        for i, v in enumerate(x_rows):
            X[i] = v
        y = self.model.predict(X) if len(X.shape) == 2 else self.model.predict([X])
        for tag_index, tag in enumerate(page_tags):
            if tag.id == "pad_tag":
                continue
            predicted_tag_types.append(TagType(tag, TAG_TYPE_BY_INDEX[np.argmax(y[tag_index - context_size])]))

        return predicted_tag_types

    def predict(self, pdf_features: "PdfFeatures") -> List[TagType]:
        pdfalto_xml = PdfAltoXml(pdf_features)
        predicted_tag_types: List[TagType] = list()
        for page in pdf_features.pages:
            if len(page.tags) == 0:
                continue
            # predicted_tag_types = self.get_predicted_tag_types(pdfalto_xml, deepcopy(page.tags), predicted_tag_types)
            predicted_tag_types = self.get_predicted_tag_types(
                pdfalto_xml, page.tags, predicted_tag_types
            )  # 4% faster execution time

        return predicted_tag_types
