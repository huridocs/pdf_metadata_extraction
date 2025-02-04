from data.TaskType import TaskType
from data.XML import XML


class ParagraphExtractorTask(TaskType):
    key: str
    xmls: list[XML]
