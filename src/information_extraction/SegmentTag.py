import bs4

ML_CLASS_LABEL_PROPERTY = "MLCLASSLABEL"
GROUP = "GROUP"


class SegmentTag:
    def __init__(
        self,
        tag_xml: bs4.Tag,
        page_width: float,
        page_height: float,
        page_number: int,
        fonts=None,
    ):
        self.page_number: int = page_number
        self.tag = tag_xml
        self.page_width = page_width
        self.page_height = page_height

        self.top = float(self.tag["VPOS"])
        self.left = float(self.tag["HPOS"])
        self.width = float(self.tag["WIDTH"])
        self.height = float(self.tag["HEIGHT"])
        self.right = self.left + self.width
        self.bottom = self.top + self.height

        self.font = []
        self.text = ""

        if len(tag_xml.find_all("String")) > 0:
            font_id = tag_xml.find_all("String")[0]["STYLEREFS"]
            self.font = list(filter(lambda font: font.id == font_id, fonts))[0]
            self.text = " ".join([string_tag["CONTENT"] for string_tag in tag_xml.find_all("String")])
