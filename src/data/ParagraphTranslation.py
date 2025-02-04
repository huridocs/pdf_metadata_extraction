from pydantic import BaseModel


class ParagraphTranslation(BaseModel):
    language: str
    text: str
    needs_user_review: bool
