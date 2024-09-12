from copy import deepcopy
from time import sleep

from httpx import ReadTimeout, ConnectTimeout
from ml_cloud_connector.MlCloudConnector import MlCloudConnector
from ollama import Client, ResponseError
from pdf_features.Rectangle import Rectangle
from pdf_token_type_labels.TokenType import TokenType

from data.PdfDataSegment import PdfDataSegment
from extractors.pdf_to_multi_option_extractor.FilterSegmentsMethod import FilterSegmentsMethod

ip_address = MlCloudConnector().get_ip()


class OllamaTranslation(FilterSegmentsMethod):
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
        text = " ".join([x.text_content for x in self.get_first_tokens(pdf_data_segments, 1500)])


        content = f"""Please translate the following text into English. Follow these guidelines:
1. Maintain the original layout and formatting.
2. Translate all text accurately without omitting any part of the content.
3. Preserve the tone and style of the original text.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.

Here is the text to be translated:
        """
        content += "\n\n" + text
        request_trial = 0
        response_message: str = ""
        cloud_wait_time = 180

        while not response_message:
            try:
                client = Client(host=f"http://{ip_address}:11434", timeout=1000)

                response = client.chat(
                    model="aya:35b",
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                )
                response_message = response["message"]["content"]
            except ReadTimeout:
                request_trial += 1
            except (ConnectTimeout, ResponseError):
                # this error happened after ~15 hours of working
                # it first happened as connecttimeout, then i saw that server is closed
                # when i re-start it nvidia-smi was not working so ollama throw a responseerror

                # ollama._types.ResponseError: unexpected server status: llm server loading model
                print(f"Response error, instance is going to be restarted [Trial: {request_trial+1}]")
                MlCloudConnector().stop()
                sleep(cloud_wait_time)
                cloud_wait_time *= 1.5
                MlCloudConnector().start()
                sleep(30)
                request_trial += 1

            if request_trial == 3:
                return [PdfDataSegment(1, Rectangle(0, 0, 0, 0), "response error")]

        return [PdfDataSegment(1, Rectangle(0, 0, 0, 0), response_message)]
