from domain.TaskType import TaskType
from domain.XML import XML


class ParagraphExtractorTask(TaskType):
    key: str
    xmls: list[XML]
