from rich.table import Table
from rich import print

from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod


def get_results_table() -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_column(justify="right")

    grid.add_row(
        "Task__________________", "Method name_____________________________________________", "Time(m)_____", "Score__"
    )

    return grid


def get_predictions_table() -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_column(justify="right")

    grid.add_row("PDF name", "Truth", "Predictions")

    return grid


def format_list(list_strings: list[str]):
    return "\n".join(sorted(list_strings)) + "\n"


def add_prediction_row(table: Table, pdf_name: str = "", truth: list[str] = None, predictions: list[str] = None):
    if not pdf_name:
        table.add_row("", "", "", "")
        return

    table.add_row(pdf_name, format_list(truth), format_list(predictions))


def add_row(table: Table, method: PdfTopicClassificationMethod = None, time: int = 0, score: float = None):
    if not method:
        table.add_row("", "", "", "")
    else:
        table.add_row(method.task_name, method.get_name(), f"{round(time / 60, 1)}", f"{round(score, 2)}%")
    print(table)
