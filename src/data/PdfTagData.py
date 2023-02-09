from pydantic import BaseModel


class PdfTagData(BaseModel):
    text: str

    @staticmethod
    def from_text(text: str):
        return PdfTagData(text=text)

    @staticmethod
    def from_texts(texts: list[str]):
        return [PdfTagData.from_text(text) for text in texts]
