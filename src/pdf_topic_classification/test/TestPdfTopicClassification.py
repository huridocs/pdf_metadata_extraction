from unittest import TestCase

from pdf_topic_classification.PdfTopicClassificationLabeledData import PdfTopicClassificationLabeledData
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from pdf_topic_classification.pdf_topic_classification_methods.FuzzyFirstCleanLabel import FuzzyFirstCleanLabel


class TestPdfTopicClassification(TestCase):
    def test_fuzzy_first_clean_label(self):
        pdf_topic_classification_labeled_data: PdfTopicClassificationLabeledData = get_labeled_data(["cejil_countries"])[0]
        method = FuzzyFirstCleanLabel()
        method.set_parameters("test", pdf_topic_classification_labeled_data)

        train_set = pdf_topic_classification_labeled_data.pdfs_labels[: -5]
        test_set = pdf_topic_classification_labeled_data.pdfs_labels[-5:]

        method.train(train_set)
        predictions = method.predict(test_set)

        self.assertEqual(len(predictions), 5)
        self.assertEqual(predictions[0], test_set[0].labels)
        self.assertEqual(predictions[-1], test_set[-1].labels)
