import json
import subprocess
from os import listdir, makedirs
from os.path import join, exists
from shutil import copy2
from pdf_features.PdfFeatures import PdfFeatures

from performance_report import get_multi_option_benchmark_data


def is_pdf_encrypted(pdf_path):
    try:
        result = subprocess.run(["qpdf", "--show-encryption", pdf_path], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return False
    return False if "File is not encrypted" in result.stdout else True


def create_xmls(pdfs_base_path: str, destination_base_path: str):
    for pdf_name in sorted(listdir(pdfs_base_path)):
        pdf_dir_name = pdf_name.replace(".pdf", "")
        pdf_destination_dir_path = join(destination_base_path, pdf_dir_name)
        makedirs(pdf_destination_dir_path, exist_ok=True)
        pdf_source_path = join(pdfs_base_path, pdf_name)
        pdf_destination_path = join(pdf_destination_dir_path, "document.pdf")
        copy2(pdf_source_path, pdf_destination_path)
        xml_path = join(pdf_destination_dir_path, "etree.xml")
        subprocess.run(["pdftohtml", "-i", "-xml", "-zoom", "1.0", pdf_destination_path, xml_path])
        if not PdfFeatures.contains_text(xml_path):
            subprocess.run(["pdftohtml", "-i", "-hidden", "-xml", "-zoom", "1.0", pdf_destination_path, xml_path])


def decrypt_pdfs(pdfs_base_path: str, filter_pdfs_path: str = ""):
    decrypted_pdf_count = 0
    filtered_pdf_names = [file.replace(".pdf", "") for file in listdir(filter_pdfs_path)] if filter_pdfs_path else []
    for pdf_name in sorted(listdir(pdfs_base_path)):
        if filtered_pdf_names and pdf_name not in filtered_pdf_names:
            continue
        xml_path = join(pdfs_base_path, pdf_name, "etree.xml")
        if exists(xml_path):
            continue
        pdf_path = join(pdfs_base_path, pdf_name, "document.pdf")
        if is_pdf_encrypted(pdf_path):
            print(f"PDF decrypted: {pdf_path}")
            subprocess.run(["qpdf", "--decrypt", "--replace-input", pdf_path])
            subprocess.run(["pdftohtml", "-i", "-xml", "-zoom", "1.0", pdf_path, xml_path])
            if not PdfFeatures.contains_text(xml_path):
                subprocess.run(["pdftohtml", "-i", "-hidden", "-xml", "-zoom", "1.0", pdf_path, xml_path])
            decrypted_pdf_count += 1
    print(f"{decrypted_pdf_count} PDFs have been decrypted.")


def check_options(task_path: str):
    with open(join(task_path, "labels.json"), mode="r") as file:
        labels_dict: dict[str, list[str]] = json.load(file)
    print(labels_dict)
    options = set([label.lower() for labels in  labels_dict.values() for label in labels])
    print(options)
    print(len(options))
    with open(join(task_path, "options.json"), mode="r") as file:
        options_json: list[str] = json.load(file)
    for option in options_json:
        if option not in options:
            print(f"Option {option} is not in labels")


def check_data():
    data = get_multi_option_benchmark_data(filter_by=["cyrilla_not_english_keywords"])
    print(data)
    # print(data[0])
    print("PDF count: ", len(data[0].samples))
    print(data[0].samples[0].pdf_data.pdf_features.file_name)
    print(data[0].samples[0].pdf_data.pdf_data_segments[0].text_content)
    print(data[0].samples[0].pdf_data.pdf_data_segments[0].segment_type)
    print(data[0].samples[0].pdf_data.pdf_data_segments[0].page_number)
    print(len(data[0].samples[0].pdf_data.pdf_data_segments))
