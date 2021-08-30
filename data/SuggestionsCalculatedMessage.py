from pydantic import BaseModel


class SuggestionsCalculatedMessage(BaseModel):
    tenant: str
    property_name: str
    success: bool
    error_message: str
