from rich.table import Table

from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod


def get_results_table() -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_column(justify="right")

    grid.add_row("Task", "Method name", "Time(m)", "Score")

    return grid


def get_predictions_table() -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_column(justify="right")

    grid.add_row("PDF name", "Truth", "Predictions")

    return grid


def format_list(list_strings: list[str]):
    return '\n'.join(sorted(list_strings)) + '\n'


def add_prediction_row(table: Table, pdf_name: str, truth: list[str], predictions: list[str]):
    table.add_row(pdf_name, format_list(truth), format_list(predictions))


def add_row(table: Table, method: PdfTopicClassificationMethod = None, time: int = 0, score: float = None):
    if not method:
        table.add_row("", "", "", "")
    else:
        print(method.task_name, method.get_name(), f"{round(time / 60, 1)}", f"{round(score, 2)}%")
        table.add_row(method.task_name, method.get_name(), f"{round(time / 60, 1)}", f"{round(score, 2)}%")
