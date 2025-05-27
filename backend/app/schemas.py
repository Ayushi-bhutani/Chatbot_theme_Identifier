from pydantic import BaseModel

class QueryRequest(BaseModel):
    text: str
    filter_by: dict = None