from pydantic import BaseModel


class SuggestionsCalculatedMessage(BaseModel):
    tenant: str
    template: str
    property_name: str
    success: bool
    error_message: str
