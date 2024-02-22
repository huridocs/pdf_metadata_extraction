from multi_option_extraction.methods.BertBaseMethod import BertBaseMethod
from multi_option_extraction.methods.FastTextMethod import FastTextMethod
from multi_option_extraction.methods.SetFitMethod import SetFitMethod
from multi_option_extraction.methods.TfIdfMethod import TfIdfMethod
from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod
from pdf_topic_classification.pdf_topic_classification_methods.NaiveMethod import NaiveMethod
from pdf_topic_classification.text_extraction_methods.TextAtTheBeginningMethod import TextAtTheBeginningMethod
from pdf_topic_classification.text_extraction_methods.TextAtTheEndMethod import TextAtTheEndMethod

PDF_TOPIC_CLASSIFICATION_METHODS = [
    NaiveMethod(),
    PdfTopicClassificationMethod(TextAtTheBeginningMethod, BertBaseMethod),
    PdfTopicClassificationMethod(TextAtTheBeginningMethod, FastTextMethod),
    PdfTopicClassificationMethod(TextAtTheBeginningMethod, SetFitMethod),
    PdfTopicClassificationMethod(TextAtTheBeginningMethod, TfIdfMethod),
    PdfTopicClassificationMethod(TextAtTheEndMethod, BertBaseMethod),
    PdfTopicClassificationMethod(TextAtTheEndMethod, FastTextMethod),
    PdfTopicClassificationMethod(TextAtTheEndMethod, SetFitMethod),
    PdfTopicClassificationMethod(TextAtTheEndMethod, TfIdfMethod),
]
