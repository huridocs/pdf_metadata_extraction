from data.PdfTagData import PdfTagData
from semantic_metadata_extraction.SemanticMethod import SemanticMethod
from dateparser.search import search_dates

from semantic_metadata_extraction.methods.DateParserMethod import DateParserMethod


class DateParserWithBreaksMethod(DateParserMethod):
    @staticmethod
    def get_date(pdf_tags: list[PdfTagData], languages):
        text = SemanticMethod.get_text_from_pdf_tags(pdf_tags)
        text_with_breaks = "\n".join([pdf_tag_data.text for pdf_tag_data in pdf_tags])

        try:
            dates = search_dates(text_with_breaks, languages=languages)
            dates_without_breaks = search_dates(text, languages=languages)

            if not dates:
                dates = list()

            if dates_without_breaks:
                dates.extend(dates_without_breaks)

            if not dates:
                dates = search_dates(text_with_breaks)

            return DateParserMethod.get_best_date(dates)

        except TypeError:
            return None
        except IndexError:
            return None
