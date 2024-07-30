import os
from copy import deepcopy

import torch
import transformers
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

from data.PdfDataSegment import PdfDataSegment
from extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import FilterSegmentsMethod


class LlamaSummary(FilterSegmentsMethod):

    valid_types = [TokenType.SECTION_HEADER, TokenType.TITLE, TokenType.TEXT, TokenType.LIST_ITEM]

    def get_first_tokens(self, pdf_data_segments: list[PdfDataSegment], text_length: int) -> list[PdfDataSegment]:
        total_text = ""
        filtered_segments: list[PdfDataSegment] = list()
        for pdf_data_segment in [x for x in pdf_data_segments if x.segment_type in self.valid_types]:
            pdf_data_segment_copy = self.clean_content_pdf_token(pdf_data_segment, text_length - len(total_text))

            if not pdf_data_segment_copy:
                break

            if pdf_data_segment_copy.text_content and "." == pdf_data_segment.text_content[-1]:
                pdf_data_segment_copy.text_content += "."

            total_text += " " + pdf_data_segment_copy.text_content
            filtered_segments.append(pdf_data_segment_copy)

        if not pdf_data_segments or "".join([x.text_content.strip() for x in filtered_segments]) == "":
            return [PdfDataSegment(1, Rectangle(0, 0, 0, 0), "no text")]

        return filtered_segments

    @staticmethod
    def clean_content_pdf_token(pdf_data_segment: PdfDataSegment, character_limit: int):
        if character_limit <= 0:
            return None

        pdf_data_segment.ml_label = 1
        pdf_data_segment_copy = deepcopy(pdf_data_segment)
        words = list()
        text = ""
        for word in pdf_data_segment_copy.text_content.split():
            clean_word = "".join([x for x in word if x.isalpha()])

            if len(text + " " + clean_word) > character_limit:
                break

            if clean_word:
                words.append(clean_word)
                text += " " + word

        pdf_data_segment_copy.text_content = " ".join(words)
        return pdf_data_segment_copy

    def filter_segments(self, pdf_data_segments: list[PdfDataSegment]) -> list[PdfDataSegment]:
        input_text = " ".join([x.text_content for x in self.get_first_tokens(pdf_data_segments, 1500)])

        model_id = "meta-llama/Meta-Llama-3.1-8B"

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            max_length=500,
            truncation=True,
            token=os.environ.get("HUGGINGFACE_TOKEN"),
            return_full_text=False,
        )

        response = pipeline(f"Select three sentences that captures the topic of the following document: {input_text}")

        if response and "generated_text" in response[0]:
            output_text = response[0]["generated_text"]
            if "three sentence" in output_text.split(":")[0]:
                output_text = ":".join(output_text.split(":")[1:]).strip()
        else:
            output_text = "No text"

        segment = PdfDataSegment(1, Rectangle(0, 0, 0, 0), output_text)
        segment.ml_label = 1
        return [segment]
