import json
import pickle
import subprocess
from collections import Counter
from os import listdir, makedirs
from os.path import join, exists
from pathlib import Path
from shutil import copy2
from time import time

import ollama
from joblib.testing import timeout
from ml_cloud_connector.MlCloudConnector import MlCloudConnector
from ollama import Client
from pdf_features.PdfFeatures import PdfFeatures
from tqdm import tqdm

from config import ROOT_PATH
from data.ExtractionData import ExtractionData
from data.PdfData import PdfData
from extractors.pdf_to_multi_option_extractor.filter_segments_methods.CleanBeginningDotDigits500 import \
    CleanBeginningDotDigits500
# from extractors.pdf_to_multi_option_extractor.filter_segments_methods.OllamaSummary import OllamaSummary
# from extractors.pdf_to_multi_option_extractor.filter_segments_methods.OllamaSummary import OllamaSummary
# from extractors.pdf_to_multi_option_extractor.filter_segments_methods.OllamaTranslation import OllamaTranslation
from performance_report import get_multi_option_benchmark_data
from langdetect import detect, LangDetectException


skip_documents_non_english = ['1523873014338iuel1m69obcb7r3z6m3g14i', '1523869532827lqgobaqd8cjtoxdldej24kj4i', '15181912931469l5ie22drqfwvii2yj4ipmn29', '1543834308928z6p03563ua', '1513174790689jwoxpxtwitdzsp6psdffg8pvi', '1521645871045ctdlwhcvcl1ftsguf3f4qdunmi', '1589183136596a3bpjfmg49', '1513173625354kfmqpp9ldeq3axm6pkap833di', '1520499246135v0lf6o9ku9jza06ani3rz4cxr', '1518787457963moxdjvvxcz1hxina0iy6wdn29', '1515660773924ojsikmt1zm4ae0p3ohqguv7vi', '1518795077783ojvjm4fdgf52ropmklcl92j4i', '1518788527689z4nyhgmanahsj4tniu11hsemi', '1512555540951c5u7adrar6fh0o5l8ejr1kyb9', '1519203688241fxpxmwqe1ht8ik35wf3arr7ldi', '1523874590432ycbgx5phs513rwu1htnh4cxr', '1523870020850y3ha3o34v8gdhg5rp6sthuxr', '1520606277297gp0bfqwf4gxeseahd1788semi', '15889311737060q4z27xf0xkc', '15181920531709xttdhi1qek0e723mmjqncdi', '1513338391206d883mzhjp1sclcyt7i1t51m7vi', '1588932612641r1dp74bb3tc', '151998395924984xu1iirshje2y0s0w5xez5mi', '15133395268581brcgxe6enlby16c1s59rizfr', '1516201798292i8qa7lm5nxskjcm05mptx1or', '1523870489810ppexsbezqr7hlgdvkftysc3di', '1516293537484x4ogu5idcf6g7z2ed31fqd7vi', '1518788375330nxbtm4esi1izo818xeh035wmi', '1520498549777knfx68xcx20hec1z3uqlzyqfr', '1513172538554iqdhp1hygemwpgyxiexumzpvi', '1513179278869h3al1mvvty99y0jbjfw29', '1513239053833tsijhxoq1725q9ajc3opnl8fr', '1513159551166f5f0d3y7sy85uelgeh1if6r', '1521642718169rdzlvuwlh8zsk1769t4gyzaor', '1513177305029gutzhiay0duxgnjr2s79o1or', '1588935685928ges8796c66k', '1519898206844076qimp3yj0a9uwifv5zouhaor', '1588604709997jud2q2ep5a', '1589189890009fuo5gxaxqg6', '1523871238142j75csq6962m5r4pa4fybep14i', '1521642050053620v4umy2e052p0uicea4cayvi', '15198958868121ykbshj27jr0g9zpv0don7b9', '1518787344830adiwemmz55ese77bu9nvzpvi', '1520843622166bju1c7ui73halo1zpsmols1yvi', '1589189161978ctw8ncmw7r9', '1512555559388r3q4cp56on5h2x59kj3lyp66r', '1519907030626w4oza1ssayhhlu0puikymn29', '1513337194382q8w25y6l10s6p2qllxd48ia4i', '15198983663897xxf1hxc0aumyfhsycwg0hpvi', '1518787568226u9p0i3jujm1eujg51m4o647vi', '15132397890210970v8z0kb09jp27r6zb0529', '1518788645031pdtndy8cikik5aba0eof20529', '1523529428306ajp40v1lzim155xl598r0sh5mi', '15204982825071rx85do1w6ws0x5v1swbnvcxr', '1513180061908tz5d0ybd73wjiehprb12it3xr', '15238710029390cx58n30yn3mias8qdoql1sjor', '1518169484620upxxb7jh9oihqgo18oqx5hfr', '158943961138913a4r9gys31d', '1556522508863lwaj46ikscq', '1542184412658sl42tsmzwad', '1522246356611roktwhlmiz48grd8u27bhjjor', '15187951649363zpxsv047q9n8g665ixes1c3di', '1513174090921bsfjlt7dwxttuj8v5dpjy8pvi', '1512553152934wetsssccooou3kukn4u9dx6r', '1535560695724ss7w7a1ormbh3ossm81wnrk9', '1516200633232obpj1sj07c3f2ro971z41jor', '1556544319089qaqz1prwpm', '1513242833351ib566uf3t4j4larfnbo869a4i', '1562164800647u9puvhpeosf', '15187952759204ajfr1whbe8icll8k19v3u9pb9', '1513334206959dt4xsuo6xkatg8dde4gqfr', '152042048424914t9pf46v56msew2xcsefdpldi', '1519988870455gujjt2ptiogdrj7sqmd4vaemi', '1520844649684mqwliu1tbmc7x0ophil5qxgvi', '1520849529661zrpvg7lowbvjzg434q9kvs4i', '15187949778758x9hhfdk5gz5cfygn3vqdunmi', '1591088080631nrtb7hqx7w', '1513266977433gijtj07ucy83cgwn6gzrrudi', '15198968410628fdddx5fv152mpagt1zmjwcdi', '1513242357345qk1fpszgkvc66n2lpikricnmi', '1523869143263ryyfr27k4ee5tvdty09cqh0k9', '1518787859354yb97oy26c5lw3m0lp649ara4i', '1512555556375iat4jqs24kb1bmod606gsnhfr', '15889324848186c9vx6h5w53', '1513180427070zylchyzwd5qhhmpmh57jsjor', '1521724537944tttxl4fxi3nhz6viehsdwvcxr', '1520606831864v5o1gxeq24i7v4000e0khjjor', '1513241746939dog8sxdlcslkuzd3mtsu1sjor', '1513179548933qn7uyh5m1kimqgnm3km3zyqfr', '15132654204828lvab6a1bza08henvet0xjemi', '1513178674952hpjww5gti1kl9riftcpu8fr', '1562162929900iud4l69yu8i', '1706658899283hlrbgkrbt3p', '1512553155169qcvh9nsvtbweapjprkonu3di', '15206092820030twhczr9w1j232d86lr14kuik9', '1513172055163vpfglglha6m0h05k04cmcxr']


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


def fix_labels():
    labels_path = "extractors/pdf_to_multi_option_extractor/labeled_data/cyrilla_not_english_keywords/labels.json"
    with open(labels_path, "r") as json_file:
        labels_dict: dict[str, list[str]] = json.load(json_file)

    for pdf_name in labels_dict.keys():
        for label_index, label in enumerate(labels_dict[pdf_name]):
            labels_dict[pdf_name][label_index] = label.lower()

    Path(labels_path).write_text(json.dumps(labels_dict, indent=4))

def create_options():
    labels_path = "extractors/pdf_to_multi_option_extractor/labeled_data/cyrilla_not_english_keywords/labels.json"
    options_path = "extractors/pdf_to_multi_option_extractor/labeled_data/cyrilla_not_english_keywords/options.json"
    with open(labels_path, "r") as json_file:
        labels_dict: dict[str, list[str]] = json.load(json_file)
    options: list[str] = list(set([label for key, labels in labels_dict.items() for label in labels]))
    Path(options_path).write_text(json.dumps(options, indent=4))


def check_ollama_summarization(pdf_pickle_path: str):
    with open(pdf_pickle_path, "rb") as pickle_file:
        pdf_data: PdfData = pickle.load(pickle_file)
    text = "\n".join([segment.text_content for segment in pdf_data.pdf_data_segments])

    prompt_1 = (f"You are going to receive a non-English text.\n"
               f"1. Select three sentences that summarizes the text\n"
               f"2. Translate the summarization into English\n\n"
               f"Here is the text:\n\n{text}")


    prompt_2 = f"""Please translate the following text into English. Follow these guidelines:
1. Maintain the original layout and formatting.
2. Translate all text accurately without omitting any part of the content.
3. Preserve the tone and style of the original text.
4. Do not include any additional comments, notes, or explanations in the output; provide only the translated text.

Here is the text to be translated:

{text}
"""

    prompt_3 = f"Select three sentences that captures the topic of the following document:\n\n{text}"

    start = time()
    response = ollama.chat(
        model="command-r",
        messages=[
            {
                "role": "user",
                "content": prompt_1
            }
        ],
    )
    response_finish_time = round(time() - start, 2)

    response_text = response["message"]["content"]
    print(response_text  + " (finished in " + str(response_finish_time) + "s)")


def check_filter_method(pdf_pickle_path=None):
    with open(pdf_pickle_path, "rb") as pickle_file:
        pdf_data: PdfData = pickle.load(pickle_file)
    pdf_data_segment = CleanBeginningDotDigits500.clean_content_pdf_token(pdf_data.pdf_data_segments[0], 500)
    print(pdf_data_segment.text_content)
    for a in pdf_data_segment.text_content:
        print(a, a.isalpha(), a.isdigit())


def get_text_contents():
    data = get_multi_option_benchmark_data(filter_by=["cyrilla_not_english_keywords"])[0]
    for sample in data.samples:
        text_content = "\n".join([segment.text_content for segment in sample.pdf_data.pdf_data_segments])
        file_name = sample.pdf_data.pdf_features.file_name + ".txt"
        Path(ROOT_PATH, "data", "multilingual_full_text", file_name).write_text(text_content)


def detect_languages():
    language_counts = Counter()
    data = get_multi_option_benchmark_data(filter_by=["cyrilla_not_english_keywords"])[0]
    not_detected_files = []
    for sample in data.samples:
        text_content = "\n".join([segment.text_content for segment in sample.pdf_data.pdf_data_segments])
        try:
            detected_language = detect(text_content)
            if detected_language == "de":
                print(sample.pdf_data.pdf_features.file_name)
        except LangDetectException:
            not_detected_files.append(sample.pdf_data.pdf_features.file_name)
            continue
        language_counts[detected_language] += 1
    # print(language_counts)
    # print(not_detected_files)


def cache_translation_summaries():
    data = get_multi_option_benchmark_data(filter_by=["cyrilla_not_english_keywords"])[0]
    already_created_files = listdir(join(ROOT_PATH, "data", "translated_summarized_data"))
    for sample in tqdm(data.samples):
        sample_dataset = ExtractionData(samples=[sample], options=data.options, multi_value=data.multi_value)
        if sample_dataset.samples[0].pdf_data.pdf_features.file_name in skip_documents_non_english:
            continue
        if sample_dataset.samples[0].pdf_data.pdf_features.file_name in already_created_files:
            continue
        print("file:",sample_dataset.samples[0].pdf_data.pdf_features.file_name)
        filtered_data = OllamaSummary().filter(sample_dataset)
        summary = " ".join([x.text_content for x in filtered_data.samples[0].pdf_data.pdf_data_segments])
        file_name = sample.pdf_data.pdf_features.file_name.replace(".pdf", ".txt")
        Path(ROOT_PATH, "data", "translated_summarized_data", file_name).write_text(summary)


def client_test():
    ip_address = MlCloudConnector().get_ip()
    client = Client(host=f"http://{ip_address}:11434", timeout=10000)
    response = client.chat(
        model="aya:35b",
        messages=[
            {
                "role": "user",
                "content": "hello",
            }
        ],

    )
    print(response["message"]["content"])


def cache_translations():
    data = get_multi_option_benchmark_data(filter_by=["cyrilla_not_english_keywords"])[0]
    already_created_files = listdir(join(ROOT_PATH, "data", "cyrilla_not_english_translations"))
    for sample in tqdm(data.samples):
        sample_dataset = ExtractionData(samples=[sample], options=data.options, multi_value=data.multi_value)
        if sample_dataset.samples[0].pdf_data.pdf_features.file_name in already_created_files:
            continue
        print("File: ", sample_dataset.samples[0].pdf_data.pdf_features.file_name)
        filtered_data = OllamaTranslation().filter(sample_dataset)
        translation = " ".join([x.text_content for x in filtered_data.samples[0].pdf_data.pdf_data_segments])
        file_name = sample.pdf_data.pdf_features.file_name
        Path(ROOT_PATH, "data", "cyrilla_not_english_translations", file_name).write_text(translation)


def cache_llama_summaries():
    translations_path = join(ROOT_PATH, "data", "cyrilla_not_english_translations")
    summaries_path = join(ROOT_PATH, "data", "cyrilla_not_english_summaries")
    for file in tqdm(sorted(listdir(translations_path))):
        text = Path(translations_path, file).read_text()
        response = ollama.chat(
            model="llama3.1:latest",
            messages=[
                {
                    "role": "user",
                    "content": f"Select three sentences that captures the topic of the following document:\n\n {text}",
                }
            ]
        )
        summary = response["message"]["content"]
        Path(summaries_path, file).write_text(summary)


def check_no_response_files():
    translations_path = join(ROOT_PATH, "data", "cyrilla_not_english_translations")
    for file in tqdm(sorted(listdir(translations_path))):
        text = Path(translations_path, file).read_text()
        if not text:
            print(file)
        if "response error" in text:
            print(file)


if __name__ == '__main__':
    # check_options("/home/ali/projects/pdf_metadata_extraction/src/extractors/pdf_to_multi_option_extractor/labeled_data/all_cyrilla_keywords")
    # decrypt_pdfs("/home/ali/projects/pdf-labeled-data/pdfs", "/home/ali/projects/pdf_metadata_extraction/data/cyrilla_not_english")
    # check_data()
    # fix_labels()
    # create_options()
    # check_ollama_summarization("/home/ali/projects/pdf_metadata_extraction/data/pdf_data_cache/1515662288224hyz2ip0wyvcdn71osbnj98uxr.pickle")
    # check_ollama_summarization("/home/ali/projects/pdf_metadata_extraction/data/pdf_data_cache/1512560170357yp7kek6mruiktm3ky4hiwwmi.pickle")
    # check_filter_method("/home/ali/projects/pdf_metadata_extraction/data/pdf_data_cache/1512560170357yp7kek6mruiktm3ky4hiwwmi.pickle")
    # get_text_contents()
    # detect_languages()
    # cache_translation_summaries()
    # client_test()
    # cache_translations()
    # check_no_response_files()
    cache_llama_summaries()


# non-english:
# Counter({'ar': 239, 'es': 219, 'fr': 195, 'en': 25, 'ca': 14, 'de': 11, 'ur': 10, 'pl': 4, 'fa': 3, 'vi': 1, 'cy': 1})
# ['1523873014338iuel1m69obcb7r3z6m3g14i', '1523869532827lqgobaqd8cjtoxdldej24kj4i', '15181912931469l5ie22drqfwvii2yj4ipmn29', '1543834308928z6p03563ua', '1513174790689jwoxpxtwitdzsp6psdffg8pvi', '1521645871045ctdlwhcvcl1ftsguf3f4qdunmi', '1589183136596a3bpjfmg49', '1513173625354kfmqpp9ldeq3axm6pkap833di', '1520499246135v0lf6o9ku9jza06ani3rz4cxr', '1518787457963moxdjvvxcz1hxina0iy6wdn29', '1515660773924ojsikmt1zm4ae0p3ohqguv7vi', '1518795077783ojvjm4fdgf52ropmklcl92j4i', '1518788527689z4nyhgmanahsj4tniu11hsemi', '1512555540951c5u7adrar6fh0o5l8ejr1kyb9', '1519203688241fxpxmwqe1ht8ik35wf3arr7ldi', '1523874590432ycbgx5phs513rwu1htnh4cxr', '1523870020850y3ha3o34v8gdhg5rp6sthuxr', '1520606277297gp0bfqwf4gxeseahd1788semi', '15889311737060q4z27xf0xkc', '15181920531709xttdhi1qek0e723mmjqncdi', '1513338391206d883mzhjp1sclcyt7i1t51m7vi', '1588932612641r1dp74bb3tc', '151998395924984xu1iirshje2y0s0w5xez5mi', '15133395268581brcgxe6enlby16c1s59rizfr', '1516201798292i8qa7lm5nxskjcm05mptx1or', '1523870489810ppexsbezqr7hlgdvkftysc3di', '1516293537484x4ogu5idcf6g7z2ed31fqd7vi', '1518788375330nxbtm4esi1izo818xeh035wmi', '1520498549777knfx68xcx20hec1z3uqlzyqfr', '1513172538554iqdhp1hygemwpgyxiexumzpvi', '1513179278869h3al1mvvty99y0jbjfw29', '1513239053833tsijhxoq1725q9ajc3opnl8fr', '1513159551166f5f0d3y7sy85uelgeh1if6r', '1521642718169rdzlvuwlh8zsk1769t4gyzaor', '1513177305029gutzhiay0duxgnjr2s79o1or', '1588935685928ges8796c66k', '1519898206844076qimp3yj0a9uwifv5zouhaor', '1588604709997jud2q2ep5a', '1589189890009fuo5gxaxqg6', '1523871238142j75csq6962m5r4pa4fybep14i', '1521642050053620v4umy2e052p0uicea4cayvi', '15198958868121ykbshj27jr0g9zpv0don7b9', '1518787344830adiwemmz55ese77bu9nvzpvi', '1520843622166bju1c7ui73halo1zpsmols1yvi', '1589189161978ctw8ncmw7r9', '1512555559388r3q4cp56on5h2x59kj3lyp66r', '1519907030626w4oza1ssayhhlu0puikymn29', '1513337194382q8w25y6l10s6p2qllxd48ia4i', '15198983663897xxf1hxc0aumyfhsycwg0hpvi', '1518787568226u9p0i3jujm1eujg51m4o647vi', '15132397890210970v8z0kb09jp27r6zb0529', '1518788645031pdtndy8cikik5aba0eof20529', '1523529428306ajp40v1lzim155xl598r0sh5mi', '15204982825071rx85do1w6ws0x5v1swbnvcxr', '1513180061908tz5d0ybd73wjiehprb12it3xr', '15238710029390cx58n30yn3mias8qdoql1sjor', '1518169484620upxxb7jh9oihqgo18oqx5hfr', '158943961138913a4r9gys31d', '1556522508863lwaj46ikscq', '1542184412658sl42tsmzwad', '1522246356611roktwhlmiz48grd8u27bhjjor', '15187951649363zpxsv047q9n8g665ixes1c3di', '1513174090921bsfjlt7dwxttuj8v5dpjy8pvi', '1512553152934wetsssccooou3kukn4u9dx6r', '1535560695724ss7w7a1ormbh3ossm81wnrk9', '1516200633232obpj1sj07c3f2ro971z41jor', '1556544319089qaqz1prwpm', '1513242833351ib566uf3t4j4larfnbo869a4i', '1562164800647u9puvhpeosf', '15187952759204ajfr1whbe8icll8k19v3u9pb9', '1513334206959dt4xsuo6xkatg8dde4gqfr', '152042048424914t9pf46v56msew2xcsefdpldi', '1519988870455gujjt2ptiogdrj7sqmd4vaemi', '1520844649684mqwliu1tbmc7x0ophil5qxgvi', '1520849529661zrpvg7lowbvjzg434q9kvs4i', '15187949778758x9hhfdk5gz5cfygn3vqdunmi', '1591088080631nrtb7hqx7w', '1513266977433gijtj07ucy83cgwn6gzrrudi', '15198968410628fdddx5fv152mpagt1zmjwcdi', '1513242357345qk1fpszgkvc66n2lpikricnmi', '1523869143263ryyfr27k4ee5tvdty09cqh0k9', '1518787859354yb97oy26c5lw3m0lp649ara4i', '1512555556375iat4jqs24kb1bmod606gsnhfr', '15889324848186c9vx6h5w53', '1513180427070zylchyzwd5qhhmpmh57jsjor', '1521724537944tttxl4fxi3nhz6viehsdwvcxr', '1520606831864v5o1gxeq24i7v4000e0khjjor', '1513241746939dog8sxdlcslkuzd3mtsu1sjor', '1513179548933qn7uyh5m1kimqgnm3km3zyqfr', '15132654204828lvab6a1bza08henvet0xjemi', '1513178674952hpjww5gti1kl9riftcpu8fr', '1562162929900iud4l69yu8i', '1706658899283hlrbgkrbt3p', '1512553155169qcvh9nsvtbweapjprkonu3di', '15206092820030twhczr9w1j232d86lr14kuik9', '1513172055163vpfglglha6m0h05k04cmcxr']



# english:
# Counter({'en': 1385, 'fr': 2, 'hu': 2, 'cy': 1, 'ar': 1})
# ['1599162321745kd1uykew87r', '1591341477329qtkrfqot41', '159916412848373l19fp0bpd', '1589446604210sexmnv6zxdt', '15887680218635ghx2mw49z', '1589534321673o5j6n9k4fpg', '158868558509864alptmkdo6']
