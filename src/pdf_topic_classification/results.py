from rich.table import Table

from pdf_topic_classification.PdfTopicClassificationMethod import PdfTopicClassificationMethod


def get_results_table() -> Table:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_column(justify="right")

    grid.add_row("Task", "Method name", "Time", "Score")

    return grid


def add_row(table: Table, method: PdfTopicClassificationMethod = None, time: int = 0, score: float = None):
    if not method:
        table.add_row("", "", "", "")
    else:
        table.add_row(method.task_name, method.get_name(), f"{round(time / 60, 1)}mins", f"{round(score, 2)}%")
