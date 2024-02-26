from os.path import join
from pathlib import Path

from paragraph_extraction_trainer.PdfSegment import PdfSegment
from pdf_token_type_labels.TokenType import TokenType
from tqdm import tqdm

from config import ROOT_PATH
from pdf_topic_classification.pdf_topic_classification_data import get_labeled_data
from transformers import pipeline

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cuda:0")
summarizer = pipeline("summarization", model="Falconsai/text_summarization", device="cuda:0")

valid_types = [TokenType.TITLE, TokenType.TEXT, TokenType.LIST]


def get_text(texts: list[str]) -> str:
    total_text = ""
    final_texts: list[str] = list()
    for text in texts:
        if len(total_text + ". " + text) > 1000:
            break

        total_text += ". " + text
        final_texts.append(text)

    return ". ".join(final_texts)


def get_summaries():
    task_labeled_data = get_labeled_data("cyrilla")[0]
    for pdf_labels in tqdm(task_labeled_data.pdfs_labels):
        path = Path(join(ROOT_PATH, "data", "summaries", f"{pdf_labels.pdf_name}.txt"))
        pdf_segments = [PdfSegment.from_pdf_tokens(paragraph.tokens) for paragraph in pdf_labels.paragraphs]
        pdf_segments = [x for x in pdf_segments if x.segment_type in valid_types]
        texts = [pdf_segment.text_content for pdf_segment in pdf_segments if pdf_segment.segment_type in valid_types]
        summary = summarizer(get_text(texts), max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        path.write_text(summary)
        print(path)


if __name__ == "__main__":
    get_summaries()
