from pydantic import BaseModel


class AbstractModel(BaseModel):
    content: str
