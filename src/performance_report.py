import json
import os
import pickle
import random
from os import listdir
from os.path import join
from pathlib import Path
from time import sleep, time

import requests
from pdf_features.PdfFeatures import PdfFeatures
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import APP_PATH, ROOT_PATH
from data.ExtractionData import ExtractionData
from data.ExtractionIdentifier import ExtractionIdentifier
from data.LabeledData import LabeledData
from data.Option import Option
from data.PdfData import PdfData
from data.PredictionSample import PredictionSample
from data.SegmentBox import SegmentBox
from data.SegmentationData import SegmentationData
from data.TrainingSample import TrainingSample
from extractors.pdf_to_multi_option_extractor.PdfMultiOptionMethod import PdfMultiOptionMethod
from extractors.pdf_to_multi_option_extractor.PdfToMultiOptionExtractor import PdfToMultiOptionExtractor

PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH = join(
    Path(__file__).parent, "extractors", "pdf_to_multi_option_extractor", "labeled_data"
)
PDF_DATA_FOLDER_PATH = join(ROOT_PATH, "data", "pdf_data_cache")
LABELED_DATA_PATH = join(APP_PATH, "pdf_topic_classification", "labeled_data")

LABELED_DATA_PDFS_PATH = join(ROOT_PATH.parent, "pdf-labeled-data", "pdfs")

BASE_LINE = {
    "cejil_president": (100.0, "NextWordsTokenSelectorFuzzy75"),
    "cyrilla_keywords": (53.49, "FuzzyFirstCleanLabel"),
    "cejil_date": (20.83, "FuzzyAll88"),
    "cejil_countries": (69.05, "FuzzyFirstCleanLabel"),
    "d4la_document_type": (44.07, "CleanBeginningDotDigits500_SingleLabelSetFit"),
    "cejil_secretary": (80.0, "FuzzyAll75"),
    "countries_in_favor": (96.89, "PreviousWordsSentenceSelectorFuzzyCommas"),
    "cejil_judge": (92.86, "FuzzyLast"),
}


def get_task_pdf_names():
    task_pdf_names: dict[str, set[str]] = dict()

    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
            labels_dict: dict[str, list[str]] = json.load(file)
            task_pdf_names.setdefault(task_name, set()).update(labels_dict.keys())

    return task_pdf_names


def cache_pdf_data(pdf_name: str, pickle_path: Path):
    pdf_features = PdfFeatures.from_poppler_etree(join(LABELED_DATA_PDFS_PATH, pdf_name, "etree.xml"))

    with open(join(LABELED_DATA_PDFS_PATH, pdf_name, "document.pdf"), "rb") as stream:
        files = {"file": stream}

        results = requests.post("http://localhost:5060", files=files)

    if results.status_code != 200:
        raise Exception("Error extracting the paragraphs")

    segments: list[SegmentBox] = [SegmentBox(**segment_box) for segment_box in results.json()]

    pdf_data = PdfData(pdf_features, file_name=pdf_name)
    segmentation_data = SegmentationData(
        page_width=segments[0].page_width,
        page_height=segments[0].page_height,
        xml_segments_boxes=segments,
        label_segments_boxes=[],
    )

    pdf_data.set_segments_from_segmentation_data(segmentation_data)

    os.makedirs(pickle_path.parent, exist_ok=True)
    with open(pickle_path, mode="wb") as file:
        pickle.dump(pdf_data, file)

    return pdf_data


def get_samples(task_name):
    with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "labels.json"), mode="r") as file:
        labels_dict: dict[str, list[str]] = json.load(file)

    multi_option_samples: list[TrainingSample] = list()
    skip_pdf_names = ['1523873014338iuel1m69obcb7r3z6m3g14i', '1523869532827lqgobaqd8cjtoxdldej24kj4i', '15181912931469l5ie22drqfwvii2yj4ipmn29', '1543834308928z6p03563ua', '1513174790689jwoxpxtwitdzsp6psdffg8pvi', '1521645871045ctdlwhcvcl1ftsguf3f4qdunmi', '1589183136596a3bpjfmg49', '1513173625354kfmqpp9ldeq3axm6pkap833di', '1520499246135v0lf6o9ku9jza06ani3rz4cxr', '1518787457963moxdjvvxcz1hxina0iy6wdn29', '1515660773924ojsikmt1zm4ae0p3ohqguv7vi', '1518795077783ojvjm4fdgf52ropmklcl92j4i', '1518788527689z4nyhgmanahsj4tniu11hsemi', '1512555540951c5u7adrar6fh0o5l8ejr1kyb9', '1519203688241fxpxmwqe1ht8ik35wf3arr7ldi', '1523874590432ycbgx5phs513rwu1htnh4cxr', '1523870020850y3ha3o34v8gdhg5rp6sthuxr', '1520606277297gp0bfqwf4gxeseahd1788semi', '15889311737060q4z27xf0xkc', '15181920531709xttdhi1qek0e723mmjqncdi', '1513338391206d883mzhjp1sclcyt7i1t51m7vi', '1588932612641r1dp74bb3tc', '151998395924984xu1iirshje2y0s0w5xez5mi', '15133395268581brcgxe6enlby16c1s59rizfr', '1516201798292i8qa7lm5nxskjcm05mptx1or', '1523870489810ppexsbezqr7hlgdvkftysc3di', '1516293537484x4ogu5idcf6g7z2ed31fqd7vi', '1518788375330nxbtm4esi1izo818xeh035wmi', '1520498549777knfx68xcx20hec1z3uqlzyqfr', '1513172538554iqdhp1hygemwpgyxiexumzpvi', '1513179278869h3al1mvvty99y0jbjfw29', '1513239053833tsijhxoq1725q9ajc3opnl8fr', '1513159551166f5f0d3y7sy85uelgeh1if6r', '1521642718169rdzlvuwlh8zsk1769t4gyzaor', '1513177305029gutzhiay0duxgnjr2s79o1or', '1588935685928ges8796c66k', '1519898206844076qimp3yj0a9uwifv5zouhaor', '1588604709997jud2q2ep5a', '1589189890009fuo5gxaxqg6', '1523871238142j75csq6962m5r4pa4fybep14i', '1521642050053620v4umy2e052p0uicea4cayvi', '15198958868121ykbshj27jr0g9zpv0don7b9', '1518787344830adiwemmz55ese77bu9nvzpvi', '1520843622166bju1c7ui73halo1zpsmols1yvi', '1589189161978ctw8ncmw7r9', '1512555559388r3q4cp56on5h2x59kj3lyp66r', '1519907030626w4oza1ssayhhlu0puikymn29', '1513337194382q8w25y6l10s6p2qllxd48ia4i', '15198983663897xxf1hxc0aumyfhsycwg0hpvi', '1518787568226u9p0i3jujm1eujg51m4o647vi', '15132397890210970v8z0kb09jp27r6zb0529', '1518788645031pdtndy8cikik5aba0eof20529', '1523529428306ajp40v1lzim155xl598r0sh5mi', '15204982825071rx85do1w6ws0x5v1swbnvcxr', '1513180061908tz5d0ybd73wjiehprb12it3xr', '15238710029390cx58n30yn3mias8qdoql1sjor', '1518169484620upxxb7jh9oihqgo18oqx5hfr', '158943961138913a4r9gys31d', '1556522508863lwaj46ikscq', '1542184412658sl42tsmzwad', '1522246356611roktwhlmiz48grd8u27bhjjor', '15187951649363zpxsv047q9n8g665ixes1c3di', '1513174090921bsfjlt7dwxttuj8v5dpjy8pvi', '1512553152934wetsssccooou3kukn4u9dx6r', '1535560695724ss7w7a1ormbh3ossm81wnrk9', '1516200633232obpj1sj07c3f2ro971z41jor', '1556544319089qaqz1prwpm', '1513242833351ib566uf3t4j4larfnbo869a4i', '1562164800647u9puvhpeosf', '15187952759204ajfr1whbe8icll8k19v3u9pb9', '1513334206959dt4xsuo6xkatg8dde4gqfr', '152042048424914t9pf46v56msew2xcsefdpldi', '1519988870455gujjt2ptiogdrj7sqmd4vaemi', '1520844649684mqwliu1tbmc7x0ophil5qxgvi', '1520849529661zrpvg7lowbvjzg434q9kvs4i', '15187949778758x9hhfdk5gz5cfygn3vqdunmi', '1591088080631nrtb7hqx7w', '1513266977433gijtj07ucy83cgwn6gzrrudi', '15198968410628fdddx5fv152mpagt1zmjwcdi', '1513242357345qk1fpszgkvc66n2lpikricnmi', '1523869143263ryyfr27k4ee5tvdty09cqh0k9', '1518787859354yb97oy26c5lw3m0lp649ara4i', '1512555556375iat4jqs24kb1bmod606gsnhfr', '15889324848186c9vx6h5w53', '1513180427070zylchyzwd5qhhmpmh57jsjor', '1521724537944tttxl4fxi3nhz6viehsdwvcxr', '1520606831864v5o1gxeq24i7v4000e0khjjor', '1513241746939dog8sxdlcslkuzd3mtsu1sjor', '1513179548933qn7uyh5m1kimqgnm3km3zyqfr', '15132654204828lvab6a1bza08henvet0xjemi', '1513178674952hpjww5gti1kl9riftcpu8fr', '1562162929900iud4l69yu8i', '1706658899283hlrbgkrbt3p', '1512553155169qcvh9nsvtbweapjprkonu3di', '15206092820030twhczr9w1j232d86lr14kuik9', '1513172055163vpfglglha6m0h05k04cmcxr']
    for pdf_name in tqdm(sorted(get_task_pdf_names()[task_name])):
        if pdf_name in skip_pdf_names:
            continue
        pickle_path = join(PDF_DATA_FOLDER_PATH, f"{pdf_name}.pickle")

        if Path(pickle_path).exists():
            with open(pickle_path, mode="rb") as file:
                pdf_data: PdfData = pickle.load(file)
                pdf_data.pdf_features.file_name = pdf_name
        else:
            pdf_data: PdfData = cache_pdf_data(pdf_name, Path(pickle_path))

        values = [Option(id=x, label=x) for x in labels_dict[pdf_name]]
        language_iso = "es" if "cejil" in task_name else "en"

        extraction_sample = TrainingSample(
            pdf_data=pdf_data, labeled_data=LabeledData(values=values, language_iso=language_iso)
        )
        multi_option_samples.append(extraction_sample)

    random.seed(42)
    random.shuffle(multi_option_samples)
    return multi_option_samples


def get_multi_option_benchmark_data(filter_by: list[str] = None) -> list[ExtractionData]:
    benchmark_data: list[ExtractionData] = list()
    for task_name in listdir(str(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH)):
        if filter_by and task_name not in filter_by:
            continue

        print(f"Loading task {task_name}")

        with open(join(PDF_MULTI_OPTION_EXTRACTION_LABELED_DATA_PATH, task_name, "options.json"), mode="r") as file:
            options = [Option(id=x, label=x) for x in json.load(file)]

        multi_option_samples = get_samples(task_name)
        multi_value: bool = len([sample for sample in multi_option_samples if len(sample.labeled_data.values) > 1]) != 0
        extraction_identifier = ExtractionIdentifier(run_name="benchmark", extraction_name=task_name)
        benchmark_data.append(
            ExtractionData(
                samples=multi_option_samples,
                options=options,
                multi_value=multi_value,
                extraction_identifier=extraction_identifier,
            )
        )

    return benchmark_data


def performance_report():
    f1s_method_name = get_f1_scores_method_names()
    sleep(1)
    print()
    print("REPORT:")
    print("-------")
    for key, (value, method_name) in f1s_method_name.items():
        if value < BASE_LINE[key][0]:
            print(f"{key}: PERFORMANCE DECREASED!!!!!")
        else:
            print(f"{key}: Good performance")

        print(f"Base performance: {BASE_LINE[key][0]}% with method {BASE_LINE[key][1]}")
        print(f"Performance: {value}% with method {method_name}")
        print()


def get_f1_scores_method_names() -> dict[str, (float, str)]:
    f1s_method_name = dict()
    for dataset in get_multi_option_benchmark_data(filter_by=[]):
        truth_one_hot, prediction_one_hot, method_name, _ = get_predictions(dataset)
        f1 = round(100 * f1_score(truth_one_hot, prediction_one_hot, average="micro"), 2)
        f1s_method_name[dataset.extraction_identifier.extraction_name] = (f1, method_name)

    return f1s_method_name


def get_predictions(dataset: ExtractionData) -> (list[list[int]], list[list[int]], str):
    training_samples_number = int(len(dataset.samples) * 0.5) if len(dataset.samples) > 10 else 10
    training_samples = dataset.samples[:training_samples_number]
    test_samples = dataset.samples[training_samples_number:] if len(dataset.samples) > 20 else dataset.samples

    training_dataset = ExtractionData(
        samples=training_samples,
        options=dataset.options,
        multi_value=dataset.multi_value,
        extraction_identifier=dataset.extraction_identifier,
    )
    extractor = PdfToMultiOptionExtractor(dataset.extraction_identifier)
    extractor.create_model(training_dataset)
    prediction_samples = [PredictionSample(pdf_data=sample.pdf_data) for sample in test_samples]
    context_samples, predictions = extractor.get_predictions(prediction_samples)
    values_list = [x.labeled_data.values for x in test_samples]
    truth_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(values_list, dataset.options)
    prediction_one_hot = PdfMultiOptionMethod.one_hot_to_options_list(predictions, dataset.options)
    return truth_one_hot, prediction_one_hot, extractor.get_predictions_method().get_name(), context_samples


def get_mistakes() -> dict[str, (float, str)]:
    f1s_method_name = dict()
    for dataset in get_multi_option_benchmark_data(filter_by=["cejil_judge"]):
        truth_one_hot, prediction_one_hot, method_name, test_samples = get_predictions(dataset)

        correct = 0
        mistakes = 0
        for truth, prediction, sample in zip(truth_one_hot, prediction_one_hot, test_samples):
            text = " ".join([x.text_content for x in sample.pdf_data.pdf_data_segments if x.ml_label])
            missing = [dataset.options[i].label for i in range(len(truth)) if truth[i] and not prediction[i]]
            wrong = [dataset.options[i].label for i in range(len(truth)) if not truth[i] and prediction[i]]

            if missing or wrong:
                print()
                print(f"PDF: {sample.pdf_data.file_name}")
                print(f"Text: {text}")
                print(f"Missing: {missing}")
                print(f"Wrong: {wrong}")
                mistakes += 1
            else:
                correct += 1

        print(f"\n\nCorrect predictions for: {correct} PDFs")
        print(f"Incorrect predictions for {mistakes} PDFs")

    return f1s_method_name


if __name__ == "__main__":
    start = time()
    print("start")
    performance_report()
    print("time", round(time() - start, 2), "s")
